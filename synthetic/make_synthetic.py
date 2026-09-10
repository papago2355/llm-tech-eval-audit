"""make_synthetic.py — seeded generator of a mathematically representative synthetic dataset.

The real linked dataset (firms evaluated between 2018 and 2023, linked to registry operating status as
of end-June 2023) is proprietary. This script produces three CSV files with the
schema documented in SCHEMA.md whose *statistical structure* is calibrated to the manuscript
(TABLE 1 composition, tier-specific failure rates, sector heterogeneity, score distributions of the
five backbones, within-tier outcome-validity ceiling, expert-grade ordinal, test-retest reliability
structure, and the two-sided screener), but which contains no real firm, no real description and no
real score.

Generative model (all latents standard normal unless stated)
------------------------------------------------------------
  z   ~ N(0,1)  latent failure risk            (the only failure-relevant signal)
  h   ~ N(0,1)  perceived technology quality  (seen by the LLMs and the expert; NOT predictive)
  q   ~ N(0,1)  description verbosity          (drives length; weakly seen by the LLMs; NOT predictive)
  u_k ~ N(0, sector_sd^2)  sector (2-digit division) effect on the failure logit
  fail | tier, z, k  ~  logistic(alpha_tier + beta*z + u_k), realised so that the failure COUNT in each
                        tier equals TABLE 1 exactly (30 / 432 / 67)
  expert grade13   : within-tier ordinal cut of  rho_g*z + lam_gh*h + noise  at the empirical
                     within-tier grade shares (higher latent -> riskier grade)
  Sonnet score     : affine( rho_s*z + lam_sh*h + lam_sq*q + noise + tier offset + sector bias v_k ),
                     clipped to [0,100], rounded — rho_s is calibrated so that within-B-tier AUROC on the
                     frozen set hits its target
  other backbones  : a_m * Sonnet_std + b_m * z + c_m * noise — (a_m, b_m) solved so that the Spearman
                     correlation with Sonnet and the Below-A AUROC hit their targets
  score shape      : every backbone latent is finally passed through a monotone sinh-arcsinh warp solved
                     so that the fail-verdict rate (share >= 50) matches TABLE 6; being monotone, the
                     warp leaves the calibrated AUROCs and Spearman correlations untouched
  TF-IDF OOF score : rho_t*z + small loadings on h, q + noise, defined only on the Below-A frozen set
  description len  : log-normal-ish, inside 80..2000 chars for exactly the per-cell counts of TIER_USABLE
                     (~81% of firms), weakly loaded on z and q
  reliability.csv  : 139 firms (25 per tier x failure cell; 14 C-tier normals), 10 reps at T=0.7 and 2 reps
                     at T=0 around the firm's Sonnet score, with a share of firms perfectly identical;
                     the firm draw is score-spread-weighted and the rep noise is calibrated so that the
                     between-firm SD and both ICC(1) values match the manuscript
  screener.csv     : 230 firms (90 A / 100 B / 40 C) with 0-100 quality scores (higher = better) for the bare
                     and the anchor-primed screener, calibrated to TABLE 7 AUROCs

Usage
-----
  python synthetic/make_synthetic.py --seed 42 --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, truncnorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import auroc, icc_oneway  # noqa: E402

# ----------------------------------------------------------------------------- constants (calibrated to the paper)
GRADE13 = ["AAA", "AA", "A+", "A", "BBB+", "BBB", "BB+", "BB", "B+", "B", "CCC", "CC", "C"]
TIER_OF_INDEX = {i: ("A" if i <= 3 else "B" if i <= 9 else "C") for i in range(13)}
TIER_INDEX_RANGE = {"A": (0, 3), "B": (4, 9), "C": (10, 12)}

TIER_N = {"A": 1466, "B": 8370, "C": 90}            # TABLE 1 (total 9,926; Below-A 8,460)
TIER_FAIL = {"A": 30, "B": 432, "C": 67}            # TABLE 1 (2.0% / 5.2% / 74.4%; total 529, Below-A 499)
# relative within-tier grade shares (aggregate counts from the source corpus; no firm-level information)
GRADE_SHARES = {"A": [79, 238, 411, 738], "B": [626, 1034, 816, 1316, 846, 3732], "C": [37, 29, 24]}
# firms retained by the 80-2,000-character description filter: (firms, failures) per tier.
# -> Below-A usable 6,774 firms / 377 failures; 210 A-tier, 1,659 B-tier and 27 C-tier firms are filtered out.
TIER_USABLE = {"A": (1256, 26), "B": (6711, 328), "C": (63, 49)}
# exact usable count per (tier, failure) cell, so that the post-filter table reproduces exactly
USABLE_EXACT = {(t, 1): f for t, (_, f) in TIER_USABLE.items()}
USABLE_EXACT.update({(t, 0): n - f for t, (n, f) in TIER_USABLE.items()})
FROZEN_B_NORMALS = 550                               # + 14 C-tier normals = 564 registry-normal firms
LEN_LOG_MEAN, LEN_LOG_SD = 5.67, 0.75               # log(chars) within the usable range
SHORT_SHARE_OF_UNUSABLE = 0.72                       # rest of the unusable descriptions are > 2000 chars

# TABLE 6 style profiles on the frozen set: mean, SD, Spearman rho vs Sonnet, Below-A AUROC target and
# fail_rate = share of prob_fail >= 50. The fail rate is hit by a monotone reshaping of the score
# distribution (shape_to_fail_rate), which leaves the AUROC and the Spearman correlations untouched.
MODELS = {
    "sonnet":   dict(mean=55.5, sd=12.6, rho_sonnet=None, auroc=0.604, fail_rate=0.70),
    "gpt55":    dict(mean=50.0, sd=13.4, rho_sonnet=0.79, auroc=0.600, fail_rate=0.49),
    "qwen35":   dict(mean=52.6, sd=21.7, rho_sonnet=0.75, auroc=0.600, fail_rate=0.55),
    "deepseek": dict(mean=59.6, sd=17.3, rho_sonnet=0.62, auroc=0.550, fail_rate=0.73),
    "llama":    dict(mean=48.1, sd=23.1, rho_sonnet=0.43, auroc=0.530, fail_rate=0.48),
}
TARGET_SONNET_WITHIN_B = 0.588     # frozen B-tier (n = 878)
TARGET_EXPERT_WITHIN_B = 0.573     # frozen B-tier
TARGET_TFIDF_BELOW_A = 0.608       # frozen Below-A (n = 941)
TARGET_LENGTH_BELOW_A = 0.524
SONNET_TIER_OFFSET = {"A": -0.90, "B": 0.0, "C": 0.60}   # in core-latent SD units (A-tier scored lower, C higher)
LAM = dict(sh=0.30, sq=0.25, gh=0.35, th=0.10, tq=0.15, lq=0.60)  # loadings on the non-predictive factors

RELIABILITY = dict(n_per_cell=25, reps_hot=10, temp_hot=0.7, identical_hot=0.58,
                   reps_cold=2, temp_cold=0.0, identical_cold=0.94,
                   between_firm_sd=15.0,       # manuscript: SD of the firm means in the reliability sample
                   icc_hot=0.986, icc_cold=0.995,   # manuscript ICC(1); the rep-noise SDs are solved for these
                   sd_hot_lo=0.5, sd_hot_hi=8.0, sd_cold_lo=0.5, sd_cold_hi=12.0)
SCREENER = dict(n_A=90, n_B=100, n_C=40, mean=55.0, sd=15.0,
                targets={"bare": (0.680, 0.636), "anchor": (0.713, 0.708)})


# ----------------------------------------------------------------------------- helpers
def bisect(fn, target, lo, hi, iters=40):
    """Monotone-increasing root finder for fn(x) = target on [lo, hi]."""
    flo, fhi = fn(lo), fn(hi)
    if not (flo <= target <= fhi):
        return lo if abs(flo - target) < abs(fhi - target) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def unit(*terms):
    """Sum of (loading, standard-normal vector) terms, completed with independent noise to unit variance."""
    load2 = sum(l * l for l, _ in terms)
    if load2 > 1:
        raise ValueError(f"loadings exceed unit variance: {load2:.3f}")
    return load2


def compose(terms, noise):
    load2 = unit(*terms)
    return sum(l * v for l, v in terms) + np.sqrt(1 - load2) * noise


def affine_to(t, mask, mean, sd, exact=False):
    """Affine map so that t[mask] has the given mean/sd; then clip to [0,100] and round.

    With exact=True the location and scale are solved so that the *realised* (rounded and clipped)
    scores carry the requested mean and SD — clipping at 0/100 otherwise shrinks the spread of the
    wide, off-centre backbone profiles by a few tenths of a point.
    """
    z = (t - t[mask].mean()) / t[mask].std()
    if not exact:
        return np.clip(np.round(mean + sd * z), 0, 100)

    def emit(m, s):
        return np.clip(np.round(m + s * z), 0, 100)

    m, s = mean, sd
    for _ in range(3):
        s = bisect(lambda x: float(emit(m, x)[mask].std(ddof=1)), sd, 0.3 * sd, 4.0 * sd, iters=22)
        m += mean - float(emit(m, s)[mask].mean())
    return emit(m, s)


def shape_to_fail_rate(t, mask, prof, threshold=50.0, lo=-1.2, hi=1.2):
    """Reshape a latent with a sinh-arcsinh warp so that the share of scores >= `threshold` on `mask`
    matches the manuscript's fail-verdict rate, then map it to the target mean/SD.

    The warp is strictly monotone, so every rank statistic already calibrated on this latent (AUROC,
    Spearman vs Sonnet) is unchanged; only the shape of the distribution between the fixed mean and SD
    moves. Returns (scores, solved skew).
    """
    a = np.arcsinh((t - t[mask].mean()) / t[mask].std())

    def score(s):
        return affine_to(np.sinh(a - s), mask, prof["mean"], prof["sd"], exact=True)

    s = bisect(lambda s_: float((score(s_)[mask] >= threshold).mean()), prof["fail_rate"], lo, hi, iters=30)
    return score(s), s


def trunc_lognormal(u, lo, hi, mu, sigma):
    """Map uniform u in (0,1) to a log-normal truncated to [lo, hi] chars."""
    a, b = (np.log(lo) - mu) / sigma, (np.log(hi) - mu) / sigma
    return np.exp(truncnorm.ppf(u, a, b, loc=mu, scale=sigma))


# ----------------------------------------------------------------------------- generator
def generate(seed: int, beta: float = 1.0, sector_sd: float = 0.4, n_sectors: int = 100,
             llm_sector_bias_sd: float = 0.25, verbose: bool = True):
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)
    report = {"seed": seed, "beta": beta, "sector_sd": sector_sd, "n_sectors": n_sectors,
              "llm_sector_bias_sd": llm_sector_bias_sd}

    # -- tiers, sectors, latents ------------------------------------------------------------
    tier = np.concatenate([np.repeat(t, n) for t, n in TIER_N.items()])
    N = len(tier)
    perm = rng.permutation(N)
    tier = tier[perm]
    sector_w = rng.gamma(0.6, size=n_sectors)
    sector_w /= sector_w.sum()
    sector_idx = rng.choice(n_sectors, size=N, p=sector_w)
    ksic2 = np.array([f"{i:02d}" for i in range(n_sectors)])[sector_idx]
    u_sector = rng.normal(0, sector_sd, n_sectors)
    v_sector = rng.normal(0, llm_sector_bias_sd, n_sectors)  # LLM sector bias, independent of risk
    z = rng.normal(size=N)
    h = rng.normal(size=N)
    q = rng.normal(size=N)

    # -- failure: exact tier counts, logistic shape ------------------------------------------
    eta = beta * z + u_sector[sector_idx]
    lat = eta + rng.logistic(size=N)
    fail = np.zeros(N, int)
    alpha = {}
    for t, nf in TIER_FAIL.items():
        idx = np.where(tier == t)[0]
        top = idx[np.argsort(lat[idx])[::-1][:nf]]
        fail[top] = 1
        alpha[t] = float(-np.sort(lat[idx])[::-1][nf - 1])  # implied intercept (threshold)
    report["implied_tier_intercepts"] = alpha

    # -- usable description flag (exact counts where the frozen set needs them) --------------
    usable = np.zeros(N, bool)
    for t in TIER_N:
        for f in (0, 1):
            idx = np.where((tier == t) & (fail == f))[0]
            k = USABLE_EXACT[(t, f)]
            usable[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True

    # -- frozen evaluation set -----------------------------------------------------------------
    frozen = np.zeros(N, bool)
    below_a_fail_usable = np.where((tier != "A") & (fail == 1) & usable)[0]
    frozen[below_a_fail_usable[:377]] = True
    frozen[np.where((tier == "C") & (fail == 0) & usable)[0]] = True
    b_norm = np.where((tier == "B") & (fail == 0) & usable)[0]
    frozen[rng.choice(b_norm, size=FROZEN_B_NORMALS, replace=False)] = True
    frozen[np.where((tier == "A") & (fail == 1) & usable)[0]] = True
    fz_below = frozen & (tier != "A")
    fz_b = frozen & (tier == "B")
    log(f"frozen set: {frozen.sum()} rows | Below-A {fz_below.sum()} ({fail[fz_below].sum()} failures) | "
        f"B-tier {fz_b.sum()} ({fail[fz_b].sum()} failures)")

    # -- description length (weakly predictive, AUROC ~0.524 on the frozen Below-A set) ------
    e_len = rng.normal(size=N)
    u_len = rng.random(N)
    short = rng.random(N) < SHORT_SHARE_OF_UNUSABLE

    def make_len(rho_l):
        w = compose([(rho_l, z), (LAM["lq"], q)], e_len)
        u = norm.cdf(w)
        L = np.empty(N)
        L[usable] = trunc_lognormal(u[usable], 80, 2000, LEN_LOG_MEAN, LEN_LOG_SD)
        m_s = ~usable & short
        m_l = ~usable & ~short
        L[m_s] = trunc_lognormal(u[m_s], 8, 79.99, 4.0, 0.6)
        L[m_l] = trunc_lognormal(u[m_l], 2001, 12000, 7.9, 0.5)  # 2001 so that floor() stays > 2000
        return np.floor(L).astype(int)

    rho_l = bisect(lambda r: auroc(fail[fz_below], make_len(r)[fz_below]), TARGET_LENGTH_BELOW_A, 0.0, 0.6)
    desc_len = make_len(rho_l)
    report["rho_length_z"] = rho_l

    # -- expert grade13 (within-tier ordinal) --------------------------------------------------
    e_g = rng.normal(size=N)

    def make_grade(rho_g):
        tg = compose([(rho_g, z), (LAM["gh"], h)], e_g)
        g = np.empty(N, int)
        for t, shares in GRADE_SHARES.items():
            idx = np.where(tier == t)[0]
            order = idx[np.argsort(tg[idx])]  # low latent -> best grade within tier
            cuts = np.cumsum(np.round(np.array(shares) / sum(shares) * len(idx))).astype(int)
            cuts[-1] = len(idx)
            lo_i = TIER_INDEX_RANGE[t][0]
            start = 0
            for j, c in enumerate(cuts):
                g[order[start:c]] = lo_i + j
                start = c
        return g

    rho_g = bisect(lambda r: auroc(fail[fz_b], make_grade(r)[fz_b]), TARGET_EXPERT_WITHIN_B, 0.0, 0.9)
    grade13 = make_grade(rho_g)
    report["rho_grade_z"] = rho_g

    # -- Sonnet score ----------------------------------------------------------------------------
    e_s = rng.normal(size=N)
    tier_off = np.vectorize(SONNET_TIER_OFFSET.get)(tier)

    def make_sonnet(rho_s):
        core = compose([(rho_s, z), (LAM["sh"], h), (LAM["sq"], q)], e_s)
        t = core + tier_off + v_sector[sector_idx]
        return affine_to(t, frozen, MODELS["sonnet"]["mean"], MODELS["sonnet"]["sd"]), t

    rho_s = bisect(lambda r: auroc(fail[fz_b], make_sonnet(r)[0][fz_b]), TARGET_SONNET_WITHIN_B, 0.0, 0.9)
    _, t_sonnet = make_sonnet(rho_s)
    score_sonnet, skew_sonnet = shape_to_fail_rate(t_sonnet, frozen, MODELS["sonnet"])
    report["rho_sonnet_z"] = rho_s
    skews = {"sonnet": skew_sonnet}
    t_s_std = (t_sonnet - t_sonnet.mean()) / t_sonnet.std()
    corr_sz = float(np.corrcoef(t_s_std, z)[0, 1])

    # -- other backbones ---------------------------------------------------------------------------
    scores = {"sonnet": score_sonnet}
    loadings = {}
    for name, prof in MODELS.items():
        if name == "sonnet":
            continue
        e_m = rng.normal(size=N)
        r_eff = prof["rho_sonnet"]  # latent correlation; nudged so the *Spearman* of the rounded scores hits the target
        for _ in range(3):
            def make_m(b, _e=e_m, _r=r_eff):
                a = _r - b * corr_sz
                var = a * a + b * b + 2 * a * b * corr_sz
                c = np.sqrt(max(1 - var, 0.0))
                t = a * t_s_std + b * z + c * _e
                return affine_to(t, frozen, MODELS[name]["mean"], MODELS[name]["sd"]), t, (a, b, c)

            b_m = bisect(lambda b: auroc(fail[fz_below], make_m(b)[0][fz_below]), prof["auroc"], -0.6, 0.9)
            sc, t_m, abc = make_m(b_m)
            rho_ach = spearmanr(score_sonnet[frozen], sc[frozen]).statistic
            r_eff = min(0.97, r_eff + (prof["rho_sonnet"] - rho_ach))
        scores[name], skews[name] = shape_to_fail_rate(t_m, frozen, prof)
        loadings[name] = dict(a=abc[0], b=abc[1], c=abc[2], latent_corr_used=r_eff)
    # DeepSeek had two unparsable outputs in the real run: mirror that so NaN handling is exercised
    nan_rows = rng.choice(np.where(frozen)[0], size=2, replace=False)
    scores["deepseek"] = scores["deepseek"].astype(float)
    scores["deepseek"][nan_rows] = np.nan
    report["backbone_loadings"] = loadings
    report["score_shape_skew"] = skews

    # -- TF-IDF out-of-fold score on the Below-A frozen rows -----------------------------------------
    e_t = rng.normal(size=N)

    def make_tfidf(rho_t):
        t = compose([(rho_t, z), (LAM["th"], h), (LAM["tq"], q)], e_t)
        return np.clip(0.48 + 0.06 * t, 0.001, 0.999)

    rho_t = bisect(lambda r: auroc(fail[fz_below], make_tfidf(r)[fz_below]), TARGET_TFIDF_BELOW_A, 0.0, 0.9)
    tfidf = make_tfidf(rho_t)
    score_tfidf = np.full(N, np.nan)
    score_tfidf[fz_below] = tfidf[fz_below]
    report["rho_tfidf_z"] = rho_t

    firm_id = np.array([f"S{i:05d}" for i in range(N)])
    firms = pd.DataFrame({
        "firm_id": firm_id, "tier": tier, "grade13": grade13, "ksic2": ksic2, "desc_len_chars": desc_len,
        "failure": fail, "score_sonnet": scores["sonnet"], "score_gpt55": scores["gpt55"],
        "score_qwen35": scores["qwen35"], "score_deepseek": scores["deepseek"], "score_llama": scores["llama"],
        "score_tfidf_oof": score_tfidf, "in_frozen_set": frozen.astype(int),
    })

    # -- reliability sample -------------------------------------------------------------------------
    # The firms are drawn per tier x failure cell with a weight that increases with |score - cell mean|;
    # the weight strength is solved so that the between-firm SD of the sample matches the manuscript.
    R = RELIABILITY

    def rel_select(lam):
        r = np.random.default_rng(seed + 101)
        picks = []
        for t in ("A", "B", "C"):
            for f in (0, 1):
                idx = np.where((tier == t) & (fail == f) & usable)[0]
                s = score_sonnet[idx]
                sd = s.std() or 1.0
                w = np.exp(lam * np.abs((s - s.mean()) / sd))
                picks.append(r.choice(idx, size=min(R["n_per_cell"], len(idx)), replace=False, p=w / w.sum()))
        return np.concatenate(picks)

    # the realised SD is a step function of the weight strength, so scan a grid instead of bisecting
    lam_grid = np.linspace(0.0, 2.0, 81)
    lam_sd = np.array([float(pd.Series(score_sonnet[rel_select(l)]).std()) for l in lam_grid])
    lam_rel = float(lam_grid[int(np.argmin(np.abs(lam_sd - R["between_firm_sd"])))])
    rel_idx = rel_select(lam_rel)
    n_rel = len(rel_idx)
    report["reliability_spread_lambda"] = lam_rel

    def make_rel(sd_hot, sd_cold):
        r = np.random.default_rng(seed + 202)
        # exact shares of firms whose re-scorings vary (the rest are byte-identical across reps)
        vary_hot = np.zeros(n_rel, bool)
        vary_hot[r.choice(n_rel, int(round((1 - R["identical_hot"]) * n_rel)), replace=False)] = True
        vary_cold = np.zeros(n_rel, bool)
        vary_cold[r.choice(n_rel, int(round((1 - R["identical_cold"]) * n_rel)), replace=False)] = True
        e_hot = r.normal(size=(n_rel, R["reps_hot"]))
        e_cold = r.normal(size=n_rel)
        rows = []
        for j, i in enumerate(rel_idx):
            mu = float(score_sonnet[i])
            for rep in range(R["reps_hot"]):
                v = float(np.clip(np.round(mu + sd_hot * e_hot[j, rep]), 0, 100)) if vary_hot[j] else mu
                rows.append((firm_id[i], R["temp_hot"], rep, v))
            rows.append((firm_id[i], R["temp_cold"], 0, mu))
            v = float(np.clip(np.round(mu + sd_cold * np.sqrt(2) * e_cold[j]), 0, 100)) if vary_cold[j] else mu
            rows.append((firm_id[i], R["temp_cold"], 1, v))
        return pd.DataFrame(rows, columns=["firm_id", "temperature", "rep", "prob_fail"])

    def icc_at(temp, df):
        g = df[df.temperature == temp]
        return icc_oneway(g["prob_fail"].values, g["firm_id"].values, n_boot=0)["icc1"]

    # ICC(1) decreases in the rep-noise SD, so bisect on its negative
    sd_hot = bisect(lambda s: -icc_at(R["temp_hot"], make_rel(s, 1.0)), -R["icc_hot"],
                    R["sd_hot_lo"], R["sd_hot_hi"], iters=25)
    sd_cold = bisect(lambda s: -icc_at(R["temp_cold"], make_rel(sd_hot, s)), -R["icc_cold"],
                     R["sd_cold_lo"], R["sd_cold_hi"], iters=25)
    reliability = make_rel(sd_hot, sd_cold)
    report["reliability_rep_sd"] = dict(hot=sd_hot, cold=sd_cold)

    # -- screener sample ------------------------------------------------------------------------------
    S = SCREENER
    scr_idx = np.concatenate([
        rng.choice(np.where((tier == "A") & usable)[0], S["n_A"], replace=False),
        rng.choice(np.where((tier == "B") & usable)[0], S["n_B"], replace=False),
        rng.choice(np.where((tier == "C") & usable)[0], S["n_C"], replace=False)])
    scr_tier = tier[scr_idx]
    isA, isC = (scr_tier == "A").astype(float), (scr_tier == "C").astype(float)
    e_shared = rng.normal(size=len(scr_idx))
    scr_scores, scr_params = {}, {}
    for variant, (tA, tC) in S["targets"].items():
        e_v = rng.normal(size=len(scr_idx))
        noise = 0.7 * e_shared + np.sqrt(1 - 0.49) * e_v

        def make_q(dA, dC):
            lat = dA * isA - dC * isC + noise
            return np.clip(np.round(S["mean"] + S["sd"] * lat / lat.std()), 0, 100)

        dA, dC = 0.5, 0.5
        for _ in range(4):  # alternating 1-D calibration
            dA = bisect(lambda d: auroc(isA, make_q(d, dC)), tA, 0.0, 3.0, iters=30)
            dC = bisect(lambda d: auroc(isC, -make_q(dA, d)), tC, 0.0, 3.0, iters=30)
        scr_scores[variant] = make_q(dA, dC)
        scr_params[variant] = dict(dA=dA, dC=dC)
    screener = pd.DataFrame({"firm_id": firm_id[scr_idx], "expert_tier": scr_tier,
                             "quality_score_bare": scr_scores["bare"],
                             "quality_score_anchor": scr_scores["anchor"]})
    report["screener_effects"] = scr_params

    # -- achieved statistics --------------------------------------------------------------------------
    ach = {}
    ach["composition"] = {t: dict(n=int((tier == t).sum()), failures=int(fail[tier == t].sum()),
                                  rate=float(fail[tier == t].mean())) for t in ("A", "B", "C")}
    in_range = (desc_len >= 80) & (desc_len <= 2000)
    ach["composition_usable"] = {t: dict(n=int((in_range & (tier == t)).sum()),
                                         failures=int(fail[in_range & (tier == t)].sum()))
                                 for t in ("A", "B", "C")}
    ach["composition_usable"]["below_A"] = dict(n=int((in_range & (tier != "A")).sum()),
                                                failures=int(fail[in_range & (tier != "A")].sum()))
    ach["length_shares"] = dict(lt80=float((desc_len < 80).mean()), usable=float(usable.mean()),
                                gt2000=float((desc_len > 2000).mean()),
                                filter_matches_flag=bool((in_range == usable).all()))
    ach["oracle_auroc"] = dict(
        z_within_B_population=auroc(fail[tier == "B"], z[tier == "B"]),
        z_below_A_frozen=auroc(fail[fz_below], z[fz_below]),
        sector_effect_within_B_population=auroc(fail[tier == "B"], u_sector[sector_idx][tier == "B"]),
    )
    ach["auroc_frozen_below_A"] = {k: auroc(fail[fz_below], v[fz_below]) for k, v in scores.items()}
    ach["auroc_frozen_below_A"].update(expert=auroc(fail[fz_below], grade13[fz_below]),
                                       tfidf=auroc(fail[fz_below], tfidf[fz_below]),
                                       length=auroc(fail[fz_below], desc_len[fz_below]))
    ach["auroc_frozen_B"] = {k: auroc(fail[fz_b], v[fz_b]) for k, v in scores.items()}
    ach["auroc_frozen_B"].update(expert=auroc(fail[fz_b], grade13[fz_b]), tfidf=auroc(fail[fz_b], tfidf[fz_b]))
    ach["auroc_population"] = dict(
        expert_full=auroc(fail, grade13), expert_below_A=auroc(fail[tier != "A"], grade13[tier != "A"]),
        expert_B=auroc(fail[tier == "B"], grade13[tier == "B"]),
        sonnet_full=auroc(fail, scores["sonnet"]), sonnet_below_A=auroc(fail[tier != "A"], scores["sonnet"][tier != "A"]),
        sonnet_B=auroc(fail[tier == "B"], scores["sonnet"][tier == "B"]),
        length_full=auroc(fail[desc_len > 0], desc_len[desc_len > 0]))
    ach["style_frozen"] = {k: dict(mean=float(np.nanmean(v[frozen])), sd=float(np.nanstd(v[frozen], ddof=1)),
                                   fail_rate=float(np.nanmean(v[frozen] >= 50)),
                                   rho_sonnet=float(spearmanr(scores["sonnet"][frozen], v[frozen], nan_policy="omit").statistic))
                           for k, v in scores.items()}
    ach["spearman_frozen_below_A"] = dict(
        sonnet_expert=float(spearmanr(scores["sonnet"][fz_below], grade13[fz_below]).statistic),
        sonnet_tfidf=float(spearmanr(scores["sonnet"][fz_below], tfidf[fz_below]).statistic),
        sonnet_length=float(spearmanr(scores["sonnet"][fz_below], desc_len[fz_below]).statistic),
        tfidf_expert=float(spearmanr(tfidf[fz_below], grade13[fz_below]).statistic))
    rel_stats = {}
    for temp, g in reliability.groupby("temperature"):
        r = icc_oneway(g["prob_fail"].values, g["firm_id"].values, n_boot=200, seed=seed)
        rel_stats[str(temp)] = dict(icc1=r["icc1"], icc1k=r["icc1k"], sem=r["sem"], n_firms=r["n_groups"],
                                    between_firm_sd=float(g.groupby("firm_id")["prob_fail"].mean().std()),
                                    identical=float((g.groupby("firm_id")["prob_fail"].nunique() == 1).mean()))
    ach["reliability"] = rel_stats
    ach["screener"] = {v: dict(A_vs_rest=auroc(isA, scr_scores[v]), C_vs_rest=auroc(isC, -scr_scores[v]))
                       for v in scr_scores}
    report["achieved"] = ach

    if verbose:
        log(json.dumps(ach, indent=1, default=float))
    return firms, reliability, screener, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--beta", type=float, default=1.0, help="logit slope of latent risk z")
    ap.add_argument("--sector-sd", type=float, default=0.4, help="SD of sector effects on the failure logit")
    ap.add_argument("--n-sectors", type=int, default=100)
    ap.add_argument("--llm-sector-bias-sd", type=float, default=0.25,
                    help="SD (core-latent units) of risk-unrelated sector shifts in the LLM score")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    firms, reliability, screener, report = generate(args.seed, args.beta, args.sector_sd, args.n_sectors,
                                                    args.llm_sector_bias_sd, verbose=not args.quiet)
    args.out.mkdir(parents=True, exist_ok=True)
    firms.to_csv(args.out / "firms.csv", index=False)
    reliability.to_csv(args.out / "reliability.csv", index=False)
    screener.to_csv(args.out / "screener.csv", index=False)
    (args.out / "generation_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {args.out}/firms.csv ({len(firms)} rows), reliability.csv ({len(reliability)} rows), "
          f"screener.csv ({len(screener)} rows), generation_report.json")


if __name__ == "__main__":
    main()
