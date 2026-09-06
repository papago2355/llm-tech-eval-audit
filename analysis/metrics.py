"""Reusable measurement functions for the LLM technology-evaluation judge audit.

Everything here is pure numpy/scipy so that the numbers reported in the paper can be
re-derived without statsmodels/pingouin. Conventions:

* ``y`` is a 0/1 outcome vector (1 = verified failure event).
* ``s`` is a risk score; higher = riskier. NaN scores are dropped pairwise.
* AUROC is computed with average ranks (Mann-Whitney), which equals
  ``sklearn.metrics.roc_auc_score`` including its treatment of ties.
* Bootstrap CIs are percentile intervals from plain row resampling with
  ``numpy.random.default_rng(seed)`` (the paper uses 2,000 resamples, seed 0).

Functions
---------
auroc, bootstrap_auroc_ci
delong_components, delong_ci, delong_test          (Sun & Xu 2014 fast algorithm)
paired_bootstrap_diff, tost_equivalence            (paired difference CI, TOST)
icc_oneway, agreement_proportions                  (ICC(1), ICC(1,k), F & cluster-bootstrap CIs, SEM/SDC)
confusion_at_threshold, ppv_npv_at_prevalence      (operating points, base-rate re-weighting)
stratified_auroc                                   (pooled within-stratum AUROC)
select_triage_thresholds, triage_cv, triage_reweight (two-sided triage screen)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
from scipy import stats
from scipy.stats import rankdata

__all__ = [
    "auroc", "bootstrap_auroc_ci", "delong_components", "delong_ci", "delong_test",
    "paired_bootstrap_diff", "tost_equivalence", "icc_oneway", "agreement_proportions",
    "confusion_at_threshold", "ppv_npv_at_prevalence", "stratified_auroc",
    "select_triage_thresholds", "triage_cv", "triage_reweight",
]


# --------------------------------------------------------------------------- helpers
def _clean(y, *scores):
    """Cast to float arrays and drop rows where y or any score is NaN."""
    y = np.asarray(y, dtype=float)
    scores = [np.asarray(s, dtype=float) for s in scores]
    mask = ~np.isnan(y)
    for s in scores:
        mask &= ~np.isnan(s)
    return (y[mask],) + tuple(s[mask] for s in scores)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) — identical to scipy.stats.rankdata(method='average')."""
    return rankdata(x, method="average")


# --------------------------------------------------------------------------- AUROC
def auroc(y, s) -> float:
    """Mann-Whitney AUROC with ties counted 1/2. Returns NaN if one class is absent."""
    y, s = _clean(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _midrank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auroc_ci(y, s, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for AUROC (rows resampled with replacement).

    Mirrors ``verify_foundation.boot_ci`` in the analysis repo: resamples that lose a class
    are skipped (they are counted in ``n_boot_used``).
    """
    y, s = _clean(y, s)
    point = auroc(y, s)
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    boots = []
    for _ in range(n_boot):
        j = rng.choice(idx, n, replace=True)
        if 0 < y[j].sum() < n:
            boots.append(auroc(y[j], s[j]))
    lo, hi = (np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if boots else (float("nan"), float("nan")))
    return {"auroc": point, "lo": float(lo), "hi": float(hi), "n": int(n),
            "n_pos": int(y.sum()), "n_neg": int(n - y.sum()), "n_boot_used": len(boots)}


# --------------------------------------------------------------------------- DeLong
def delong_components(y, scores) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong (Sun & Xu, IEEE SPL 2014): AUROCs and their covariance matrix.

    ``scores`` is (k, n) — k correlated predictors on the same n rows. Returns
    ``(aucs, S)`` with ``S`` the k x k covariance of the AUROC estimates. Complexity
    O(k n log n) instead of the O(k m n) brute force.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(scores, dtype=float))
    if X.shape[1] != len(y):
        raise ValueError("scores must be (k, n) with n == len(y)")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("delong_components does not accept NaN; drop them first")
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("both classes required")
    k = X.shape[0]
    v01 = np.empty((k, m))  # structural components for positives  (Sun & Xu eq. 3)
    v10 = np.empty((k, n))  # structural components for negatives
    aucs = np.empty(k)
    for r in range(k):
        x_pos, x_neg = X[r, pos], X[r, neg]
        tx = _midrank(x_pos)
        ty = _midrank(x_neg)
        tz = _midrank(np.concatenate([x_pos, x_neg]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    S = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, S


def _brute_force_delong_components(y, s):
    """O(m n) reference implementation of the DeLong structural components (for tests)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    x_pos, x_neg = s[y == 1], s[y == 0]
    m, n = len(x_pos), len(x_neg)
    psi = (x_pos[:, None] > x_neg[None, :]).astype(float) + 0.5 * (x_pos[:, None] == x_neg[None, :])
    v10 = psi.mean(axis=1)  # one per positive
    v01 = psi.mean(axis=0)  # one per negative
    auc = psi.mean()
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    return auc, var, v10, v01


def delong_ci(y, s, alpha: float = 0.05) -> dict:
    """Asymptotic (DeLong) CI for a single AUROC."""
    y, s = _clean(y, s)
    aucs, S = delong_components(y, s[None, :])
    se = float(np.sqrt(S[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auroc": float(aucs[0]), "se": se, "lo": float(aucs[0] - z * se),
            "hi": float(aucs[0] + z * se), "n": int(len(y))}


def delong_test(y, s1, s2) -> dict:
    """Two-sided DeLong test for two correlated AUROCs measured on the same rows."""
    y, s1, s2 = _clean(y, s1, s2)
    aucs, S = delong_components(y, np.vstack([s1, s2]))
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    diff = float(aucs[0] - aucs[1])
    z = diff / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan")
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff, "se": se,
            "z": float(z), "p": p, "n": int(len(y))}


# --------------------------------------------------------------------------- paired difference / TOST
def paired_bootstrap_diff(y, s1, s2, n_boot: int = 4000, seed: int = 0) -> dict:
    """Paired bootstrap of AUROC(s1) - AUROC(s2) on the same resampled rows."""
    y, s1, s2 = _clean(y, s1, s2)
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        j = rng.choice(idx, n, replace=True)
        if 0 < y[j].sum() < n:
            diffs.append(auroc(y[j], s1[j]) - auroc(y[j], s2[j]))
    d = np.asarray(diffs)
    return {"diff": float(auroc(y, s1) - auroc(y, s2)), "boot_mean_diff": float(d.mean()),
            "ci95": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "ci90": (float(np.percentile(d, 5)), float(np.percentile(d, 95))),
            "n": int(n), "n_boot_used": int(len(d))}


def tost_equivalence(y, s1, s2, margin: float = 0.05, n_boot: int = 4000, seed: int = 0) -> dict:
    """Two one-sided tests for equivalence of two AUROCs within +/- ``margin``.

    Two flavours are returned:
    * bootstrap: equivalent if the 90% paired-bootstrap CI of the difference lies inside
      (-margin, +margin) — this is the procedure used in the paper;
    * DeLong: p_tost = max(p_lower, p_upper) from the asymptotic SE of the difference.
    """
    boot = paired_bootstrap_diff(y, s1, s2, n_boot=n_boot, seed=seed)
    lo90, hi90 = boot["ci90"]
    dl = delong_test(y, s1, s2)
    if dl["se"] > 0:
        p_lower = float(stats.norm.sf((dl["diff"] + margin) / dl["se"]))   # H0: diff <= -margin
        p_upper = float(stats.norm.cdf((dl["diff"] - margin) / dl["se"]))  # H0: diff >= +margin
        p_tost = max(p_lower, p_upper)
    else:
        p_lower = p_upper = p_tost = float("nan")
    return {"diff": dl["diff"], "margin": margin, "ci90_boot": (lo90, hi90),
            "equivalent_boot": bool(lo90 > -margin and hi90 < margin),
            "delong_se": dl["se"], "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "equivalent_delong_005": bool(p_tost < 0.05) if p_tost == p_tost else False,
            "n": dl["n"], "n_boot_used": boot["n_boot_used"]}


# --------------------------------------------------------------------------- ICC
def _oneway_anova(values: np.ndarray, groups: np.ndarray):
    """Return (a, n, k0, MSB, MSW) for a one-way random-effects ANOVA."""
    _, inv = np.unique(groups, return_inverse=True)
    a = int(inv.max() + 1)
    n = len(values)
    n_i = np.bincount(inv, minlength=a).astype(float)
    sum_i = np.bincount(inv, weights=values, minlength=a)
    sumsq_i = np.bincount(inv, weights=values ** 2, minlength=a)
    grand = values.mean()
    mean_i = sum_i / n_i
    ssb = float((n_i * (mean_i - grand) ** 2).sum())
    ssw = float((sumsq_i - sum_i ** 2 / n_i).sum())
    msb = ssb / (a - 1)
    msw = ssw / (n - a)
    k0 = (n - (n_i ** 2).sum() / n) / (a - 1)  # average group size for unbalanced designs
    return a, n, float(k0), msb, msw


def _icc_from_ms(msb, msw, k0):
    icc1 = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else float("nan")
    icc1k = (msb - msw) / msb if msb > 0 else float("nan")
    return float(icc1), float(icc1k)


def icc_oneway(values, groups, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict:
    """One-way random-effects ICC — ICC(1) single rating and ICC(1,k) mean of k ratings.

    Model: rating_ij = mu + firm_i + e_ij (raters/repetitions are not identifiable, which is the
    correct model for repeated scorings of the same description). CIs:
    * exact F-distribution intervals (McGraw & Wong 1996; Shrout & Fleiss 1979 case 1), using
      the average group size k0 for unbalanced designs;
    * cluster (firm-level) percentile bootstrap.
    Also returns SEM = sqrt(MSW) and SDC95 = 1.96 * sqrt(2) * SEM (smallest detectable change).
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    m = ~np.isnan(values)
    values, groups = values[m], groups[m]
    a, n, k0, msb, msw = _oneway_anova(values, groups)
    icc1, icc1k = _icc_from_ms(msb, msw, k0)
    F = msb / msw if msw > 0 else float("inf")
    df1, df2 = a - 1, n - a
    f_hi = stats.f.ppf(1 - alpha / 2, df1, df2)
    f_hi_rev = stats.f.ppf(1 - alpha / 2, df2, df1)
    FL, FU = F / f_hi, F * f_hi_rev
    icc1_f = ((FL - 1) / (FL + k0 - 1), (FU - 1) / (FU + k0 - 1))
    icc1k_f = (1 - 1 / FL, 1 - 1 / FU)

    # cluster bootstrap over firms (resampled firms treated as distinct clusters)
    _, inv = np.unique(groups, return_inverse=True)
    n_i = np.bincount(inv, minlength=a).astype(float)
    sum_i = np.bincount(inv, weights=values, minlength=a)
    sumsq_i = np.bincount(inv, weights=values ** 2, minlength=a)
    rng = np.random.default_rng(seed)
    b1, bk = [], []
    for _ in range(n_boot):
        j = rng.integers(0, a, a)
        nn, ss, sq = n_i[j], sum_i[j], sumsq_i[j]
        N = nn.sum()
        grand = ss.sum() / N
        ssb = ((nn * (ss / nn - grand) ** 2)).sum()
        ssw = (sq - ss ** 2 / nn).sum()
        if N - a <= 0:
            continue
        msb_b, msw_b = ssb / (a - 1), ssw / (N - a)
        k0_b = (N - (nn ** 2).sum() / N) / (a - 1)
        i1, ik = _icc_from_ms(msb_b, msw_b, k0_b)
        b1.append(i1)
        bk.append(ik)
    def pct(v):
        if len(v) == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
    sem = float(np.sqrt(msw))
    return {"n_groups": a, "n_obs": n, "k0": k0, "MSB": msb, "MSW": msw, "F": float(F),
            "icc1": icc1, "icc1_f_ci": tuple(map(float, icc1_f)), "icc1_boot_ci": pct(b1),
            "icc1k": icc1k, "icc1k_f_ci": tuple(map(float, icc1k_f)), "icc1k_boot_ci": pct(bk),
            "sem": sem, "sdc95": float(1.96 * np.sqrt(2) * sem), "n_boot_used": len(b1)}


def agreement_proportions(values, groups, threshold: float = 50.0,
                          within: Sequence[float] = (0, 5, 10, 20)) -> dict:
    """Rep-to-rep agreement summaries per firm: score span and verdict changes.

    * ``identical`` — share of firms whose repeated scores are all byte-identical (span 0);
    * ``within_<x>`` — share of firms whose span (max - min) is <= x points;
    * ``verdict_change_rate`` — share of firms whose repeats fall on both sides of
      ``threshold`` (the fail/survive flag would differ between re-scorings).
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    m = ~np.isnan(values)
    values, groups = values[m], groups[m]
    uniq, inv = np.unique(groups, return_inverse=True)
    a = len(uniq)
    mx = np.full(a, -np.inf)
    mn = np.full(a, np.inf)
    np.maximum.at(mx, inv, values)
    np.minimum.at(mn, inv, values)
    cnt = np.bincount(inv, minlength=a)
    ok = cnt >= 2
    span = (mx - mn)[ok]
    flagged_any = np.zeros(a, bool)
    unflagged_any = np.zeros(a, bool)
    np.logical_or.at(flagged_any, inv, values >= threshold)
    np.logical_or.at(unflagged_any, inv, values < threshold)
    change = (flagged_any & unflagged_any)[ok]
    out = {"n_firms": int(ok.sum()), "identical": float((span == 0).mean()),
           "span_median": float(np.median(span)), "span_mean": float(span.mean()),
           "span_max": float(span.max()), "verdict_change_rate": float(change.mean()),
           "threshold": threshold}
    for w in within:
        out[f"within_{int(w)}"] = float((span <= w).mean())
    return out


# --------------------------------------------------------------------------- operating points
def confusion_at_threshold(y, s, thr: float = 50.0) -> dict:
    """Confusion counts flagging ``s >= thr``. 'capture' = sensitivity, 'burden' = FPR."""
    y, s = _clean(y, s)
    flag = s >= thr
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"threshold": thr, "n": int(len(y)), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "capture": sens, "burden": 1 - spec if spec == spec else float("nan"), "specificity": spec,
            "flag_rate": float(flag.mean()),
            "ppv_sample": tp / (tp + fp) if tp + fp else float("nan"),
            "npv_sample": tn / (tn + fn) if tn + fn else float("nan"),
            "prevalence_sample": float(y.mean())}


def ppv_npv_at_prevalence(sens: float, spec: float, prev: float) -> dict:
    """Re-weight PPV/NPV to a target prevalence (Bayes) from sensitivity and specificity."""
    ppv_den = sens * prev + (1 - spec) * (1 - prev)
    npv_den = spec * (1 - prev) + (1 - sens) * prev
    return {"prevalence": prev,
            "ppv": sens * prev / ppv_den if ppv_den > 0 else float("nan"),
            "npv": spec * (1 - prev) / npv_den if npv_den > 0 else float("nan"),
            "flag_rate": ppv_den}


# --------------------------------------------------------------------------- stratified AUROC
def stratified_auroc(y, s, strata, min_per_class: int = 1) -> dict:
    """Pooled within-stratum AUROC: P(score_fail > score_normal | same stratum).

    Equals the pair-weighted average of within-stratum AUROCs (weights n1_k * n0_k) over strata
    with at least ``min_per_class`` firms of each class. Removes any between-stratum
    (e.g. sector-level) shift in score or prevalence from the comparison.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    strata = np.asarray(strata).astype(str)
    m = ~np.isnan(y) & ~np.isnan(s)
    y, s, strata = y[m], s[m], strata[m]
    num = den = 0.0
    rows = []
    for k in np.unique(strata):
        sel = strata == k
        n1, n0 = int(y[sel].sum()), int((1 - y[sel]).sum())
        if n1 >= min_per_class and n0 >= min_per_class:
            a = auroc(y[sel], s[sel])
            w = n1 * n0
            num += a * w
            den += w
            rows.append({"stratum": k, "n": int(sel.sum()), "n_pos": n1, "n_neg": n0, "auroc": a})
    return {"pooled_auroc": num / den if den else float("nan"), "n_strata_used": len(rows),
            "n_rows_used": int(sum(r["n"] for r in rows)), "n_rows_total": int(len(y)),
            "pooled_unstratified_auroc": auroc(y, s), "per_stratum": rows}


# --------------------------------------------------------------------------- triage
def select_triage_thresholds(y, s, accept_ratio: float = 0.5, reject_lift: float = 1.5,
                             min_band_n: int = 20) -> dict:
    """Choose two-sided triage cut-offs on a training sample.

    * clear-accept band = {score <= t_lo}: the largest t_lo such that the failure rate in the band
      is <= accept_ratio * prevalence and the band holds >= min_band_n firms;
    * clear-reject band = {score >= t_hi}: the smallest t_hi such that the failure rate in the band
      is >= reject_lift * prevalence and the band holds >= min_band_n firms;
    * t_lo < t_hi is enforced (bands cannot overlap). Missing bands are returned as -inf / +inf.
    """
    y, s = _clean(y, s)
    prev = y.mean()
    grid = np.unique(s)
    order = np.argsort(s, kind="mergesort")
    ys, ss = y[order], s[order]
    cum_pos = np.cumsum(ys)
    n = len(ys)
    t_lo, t_hi = -np.inf, np.inf
    # accept: scan thresholds upward
    for t in grid:
        k = int(np.searchsorted(ss, t, side="right"))  # firms with score <= t
        if k >= min_band_n and cum_pos[k - 1] / k <= accept_ratio * prev:
            t_lo = float(t)
    # reject: scan thresholds downward
    for t in grid[::-1]:
        k0 = int(np.searchsorted(ss, t, side="left"))  # first index with score >= t
        k = n - k0
        if k >= min_band_n and (cum_pos[-1] - (cum_pos[k0 - 1] if k0 > 0 else 0)) / k >= reject_lift * prev:
            t_hi = float(t)
    if t_lo >= t_hi:  # overlapping bands -> keep the reject band, drop accept
        t_lo = -np.inf
    return {"t_lo": t_lo, "t_hi": t_hi, "train_prevalence": float(prev), "n_train": int(n)}


def _route(s, t_lo, t_hi):
    s = np.asarray(s, dtype=float)
    r = np.full(len(s), "review", dtype=object)
    r[s <= t_lo] = "accept"
    r[s >= t_hi] = "reject"
    return r


def triage_cv(y, s, n_splits: int = 5, seed: int = 42, **rule) -> dict:
    """Cross-validated two-sided triage: thresholds fit on k-1 folds, routing evaluated out-of-fold.

    Returns routing proportions, false-accept (failures routed to accept) and false-reject
    (registry-normal firms routed to reject) counts, per-band failure rates, per-class routing
    rates (for base-rate re-weighting) and the per-fold thresholds.
    """
    from sklearn.model_selection import StratifiedKFold

    y, s = _clean(y, s)
    yi = y.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    route = np.empty(len(y), dtype=object)
    folds = []
    for tr, te in skf.split(s, yi):
        th = select_triage_thresholds(y[tr], s[tr], **rule)
        route[te] = _route(s[te], th["t_lo"], th["t_hi"])
        folds.append(th)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
           "rule": {"accept_ratio": rule.get("accept_ratio", 0.5), "reject_lift": rule.get("reject_lift", 1.5),
                    "min_band_n": rule.get("min_band_n", 20)},
           "folds": folds, "bands": {}}
    for band in ("accept", "review", "reject"):
        sel = route == band
        n_b, pos_b = int(sel.sum()), int(y[sel].sum())
        out["bands"][band] = {"n": n_b, "share": n_b / len(y), "n_fail": pos_b, "n_normal": n_b - pos_b,
                              "fail_rate": pos_b / n_b if n_b else float("nan"),
                              "p_band_given_fail": pos_b / y.sum() if y.sum() else float("nan"),
                              "p_band_given_normal": (n_b - pos_b) / (len(y) - y.sum())}
    out["false_accepts"] = out["bands"]["accept"]["n_fail"]
    out["false_rejects"] = out["bands"]["reject"]["n_normal"]
    out["routing"] = [str(r) for r in route]
    return out


def triage_reweight(triage: dict, prev: float) -> dict:
    """Re-weight OOF routing to a target prevalence using per-class routing rates."""
    rows = {}
    for band, b in triage["bands"].items():
        pf, pn = b["p_band_given_fail"], b["p_band_given_normal"]
        share = pf * prev + pn * (1 - prev)
        rows[band] = {"share": share, "fail_rate": (pf * prev / share) if share > 0 else float("nan")}
    rows["false_accept_rate_among_failures"] = triage["bands"]["accept"]["p_band_given_fail"]
    rows["false_reject_rate_among_normals"] = triage["bands"]["reject"]["p_band_given_normal"]
    rows["prevalence"] = prev
    return rows
