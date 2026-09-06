"""reproduce.py — re-derive the manuscript's measurement tables from firms.csv / reliability.csv / screener.csv.

Works unchanged on the synthetic data (synthetic/make_synthetic.py) and on the de-identified real
extract (SCHEMA.md). Writes <out>/tables.md, <out>/results.json and two figures.

Usage:
    python analysis/reproduce.py --data data/synthetic [--out data/synthetic/output] [--n-boot 2000]

Tables produced (numbering mirrors the manuscript):
    Table I      composition by tier (population)
    Table V-1    method plateau on the Below-A frozen set: AUROC with bootstrap 95% CI + DeLong CI
    Table V-1b   the same within the B-tier
    Table V-1c   pairwise DeLong tests and bootstrap TOST (Sonnet vs each predictor, expert vs TF-IDF)
    Table V-2    style profiles (mean, SD, fail-verdict rate, Spearman vs Sonnet) on the frozen set
    Table V-4    reliability: ICC(1), ICC(1,k) with F and cluster-bootstrap CIs, SEM, SDC, agreement
    Table OP     operating points at prob_fail >= 50, PPV/NPV re-weighted to the population prevalence
    Table SEC    sector-stratified (within-ksic2) pooled AUROC vs unstratified
    Table TRI    cross-validated two-sided triage routing
    Table V-3    two-sided screener AUROCs (bare / anchor-primed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import metrics as M  # noqa: E402

GRADE13 = ["AAA", "AA", "A+", "A", "BBB+", "BBB", "BB+", "BB", "B+", "B", "CCC", "CC", "C"]
PREDICTORS = [  # column, label
    ("grade13", "Expert grade (ordinal 0-12)"),
    ("score_tfidf_oof", "TF-IDF (char n-gram) OOF"),
    ("score_sonnet", "Sonnet 4.6"),
    ("score_gpt55", "GPT-5.5"),
    ("score_qwen35", "Qwen 3.5"),
    ("score_deepseek", "DeepSeek V4"),
    ("score_llama", "Llama 4 Maverick"),
    ("desc_len_chars", "Length-only"),
]
LLM_COLS = ["score_sonnet", "score_gpt55", "score_qwen35", "score_deepseek", "score_llama"]
LABEL = dict(PREDICTORS)
THRESHOLD = 50.0


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Report:
    def __init__(self):
        self.md = []
        self.json = {}

    def section(self, title, note=None):
        self.md += ["", f"## {title}", ""]
        if note:
            self.md += [f"*{note}*", ""]

    def table(self, header, rows):
        self.md.append(md_table(header, rows))
        self.md.append("")

    def text(self, s):
        self.md += [s, ""]


# ----------------------------------------------------------------------------- tables
def table_composition(firms, R):
    R.section("Table I — Dataset composition by grade tier (population)")
    rows, js = [], {}
    for t in ("A", "B", "C"):
        g = firms[firms.tier == t]
        rows.append([f"{t}-tier", f"{len(g):,}", f"{int(g.failure.sum()):,}", f"{100 * g.failure.mean():.1f}%"])
        js[t] = dict(n=int(len(g)), failures=int(g.failure.sum()), rate=float(g.failure.mean()))
    ba = firms[firms.tier != "A"]
    rows.append(["Below-A (B + C)", f"{len(ba):,}", f"{int(ba.failure.sum()):,}", f"{100 * ba.failure.mean():.1f}%"])
    rows.append(["Full labeled corpus", f"{len(firms):,}", f"{int(firms.failure.sum()):,}", f"{100 * firms.failure.mean():.1f}%"])
    js["below_A"] = dict(n=int(len(ba)), failures=int(ba.failure.sum()), rate=float(ba.failure.mean()))
    js["full"] = dict(n=int(len(firms)), failures=int(firms.failure.sum()), rate=float(firms.failure.mean()))
    R.table(["Grade tier", "Firms", "Failures", "Rate"], rows)
    usable = firms.desc_len_chars.between(80, 2000)
    fz = firms[firms.in_frozen_set == 1]
    by_tier = ", ".join(f"{t} {int((usable & (firms.tier == t)).sum()):,}/"
                        f"{int(firms.failure[usable & (firms.tier == t)].sum()):,}" for t in ("A", "B", "C"))
    js["usable"] = {t: dict(n=int((usable & (firms.tier == t)).sum()),
                            failures=int(firms.failure[usable & (firms.tier == t)].sum())) for t in ("A", "B", "C")}
    js["usable"]["below_A"] = dict(n=int((usable & (firms.tier != "A")).sum()),
                                   failures=int(firms.failure[usable & (firms.tier != "A")].sum()))
    js["usable"]["full"] = dict(n=int(usable.sum()), failures=int(firms.failure[usable].sum()))
    R.text(f"Usable description (80–2,000 chars): {int(usable.sum()):,} firms "
           f"({int(firms.failure[usable].sum()):,} failures) — by tier (firms/failures) {by_tier}; Below-A usable: "
           f"{int((usable & (firms.tier != 'A')).sum()):,} firms ({int(firms.failure[usable & (firms.tier != 'A')].sum()):,} failures). "
           f"Frozen evaluation set: {len(fz):,} rows = {int(fz.failure.sum())} failures "
           f"(incl. {int(((fz.tier == 'A') & (fz.failure == 1)).sum())} A-tier) + {int((fz.failure == 0).sum())} registry-normal; "
           f"Below-A rows {int((fz.tier != 'A').sum())}, B-tier rows {int((fz.tier == 'B').sum())}, C-tier rows {int((fz.tier == 'C').sum())}.")
    js["frozen"] = dict(n=int(len(fz)), failures=int(fz.failure.sum()), normals=int((fz.failure == 0).sum()),
                        below_A=int((fz.tier != "A").sum()), B=int((fz.tier == "B").sum()), C=int((fz.tier == "C").sum()))
    R.json["table_I"] = js
    return js


def table_plateau(df, R, title, key, n_boot, seed):
    R.section(title, f"n = {len(df)} rows, {int(df.failure.sum())} failures, {int((df.failure == 0).sum())} registry-normal. "
              f"Bootstrap: {n_boot} row resamples, seed {seed}, percentile 2.5/97.5. DeLong CI is asymptotic.")
    rows, js = [], {}
    for col, label in PREDICTORS:
        if col not in df or df[col].notna().sum() < 10:
            continue
        b = M.bootstrap_auroc_ci(df.failure, df[col], n_boot=n_boot, seed=seed)
        d = M.delong_ci(df.failure, df[col])
        rows.append([label, b["n"], f3(b["auroc"]), ci(b["lo"], b["hi"]), ci(d["lo"], d["hi"])])
        js[col] = dict(n=b["n"], n_pos=b["n_pos"], auroc=b["auroc"], boot_lo=b["lo"], boot_hi=b["hi"],
                       delong_se=d["se"], delong_lo=d["lo"], delong_hi=d["hi"])
    R.table(["Predictor", "n", "AUROC", "95% bootstrap CI", "95% DeLong CI"], rows)
    R.json[key] = js
    return js


def table_pairwise(df, R, title, key, pairs, n_boot, seed, margin=0.05):
    R.section(title, f"n = {len(df)}. DeLong: two-sided test for correlated AUROCs. TOST: equivalence if the 90% paired-bootstrap "
              f"CI of the AUROC difference lies within ±{margin} ({n_boot} resamples, seed {seed}); p_TOST from the DeLong SE.")
    rows, js = [], {}
    for a, b in pairs:
        if a not in df or b not in df:
            continue
        sub = df.dropna(subset=[a, b])
        dl = M.delong_test(sub.failure, sub[a], sub[b])
        ts = M.tost_equivalence(sub.failure, sub[a], sub[b], margin=margin, n_boot=n_boot, seed=seed)
        rows.append([f"{LABEL[a]} vs {LABEL[b]}", dl["n"], f3(dl["auc1"]), f3(dl["auc2"]), f"{dl['diff']:+.3f}",
                     f3(dl["se"]), f"{dl['p']:.3f}", f"[{ts['ci90_boot'][0]:+.3f}, {ts['ci90_boot'][1]:+.3f}]",
                     f"{ts['p_tost']:.3f}", "yes" if ts["equivalent_boot"] else "no"])
        js[f"{a}__vs__{b}"] = dict(n=dl["n"], auc1=dl["auc1"], auc2=dl["auc2"], diff=dl["diff"], delong_se=dl["se"],
                                   delong_p=dl["p"], tost_ci90=ts["ci90_boot"], tost_p=ts["p_tost"],
                                   equivalent_boot=ts["equivalent_boot"])
    R.table(["Pair", "n", "AUROC 1", "AUROC 2", "Δ", "SE", "DeLong p", "90% CI (Δ)", "p_TOST", "equiv. ±0.05"], rows)
    R.json[key] = js


def table_style(fz, R):
    R.section("Table V-2 — Cross-model style profiles (frozen set)",
              f"n = {len(fz)}; fail-rate = share with prob_fail ≥ {THRESHOLD:.0f}; Spearman ρ vs Sonnet on pairwise-complete rows.")
    rows, js = [], {}
    for col in LLM_COLS:
        if col not in fz:
            continue
        s = fz[col]
        rho = spearmanr(fz["score_sonnet"], s, nan_policy="omit").statistic
        rows.append([LABEL[col], int(s.notna().sum()), f"{s.mean():.1f}", f"{s.std():.1f}", f"{100 * (s >= THRESHOLD).mean():.0f}%", f"{rho:.2f}"])
        js[col] = dict(n=int(s.notna().sum()), mean=float(s.mean()), sd=float(s.std()),
                       fail_rate=float((s >= THRESHOLD).mean()), rho_vs_sonnet=float(rho))
    R.table(["Model", "n", "Mean", "SD", "Fail-rate", "ρ vs Sonnet"], rows)
    R.json["table_V2"] = js


def table_reliability(rel, firms, R, n_boot, seed):
    R.section("Table V-4 — Test–retest reliability of the judge on byte-identical descriptions",
              "One-way random-effects ICC: ICC(1) single rating, ICC(1,k) mean of k re-scorings; F-distribution CI (exact) "
              "and firm-level cluster-bootstrap CI; SEM = √MSW; SDC95 = 1.96·√2·SEM; spans are max−min over re-scorings.")
    rows, agr_rows, tier_rows, js = [], [], [], {}
    tier_of = firms.set_index("firm_id")["tier"]
    for temp, g in sorted(rel.groupby("temperature"), key=lambda kv: -kv[0]):
        g = g.dropna(subset=["prob_fail"])
        cnt = g.groupby("firm_id").size()
        g = g[g.firm_id.isin(cnt[cnt >= 2].index)]
        r = M.icc_oneway(g.prob_fail, g.firm_id, n_boot=n_boot, seed=seed)
        a = M.agreement_proportions(g.prob_fail, g.firm_id, threshold=THRESHOLD)
        rows.append([f"{temp:g}", r["n_groups"], r["n_obs"], f"{r['k0']:.1f}", f3(r["icc1"]), ci(*r["icc1_f_ci"]), ci(*r["icc1_boot_ci"]),
                     f3(r["icc1k"]), ci(*r["icc1k_f_ci"]), ci(*r["icc1k_boot_ci"]), f"{r['sem']:.2f}", f"{r['sdc95']:.1f}"])
        agr_rows.append([f"{temp:g}", a["n_firms"], f"{100 * a['identical']:.0f}%", f"{100 * a['within_5']:.0f}%", f"{100 * a['within_10']:.0f}%",
                         f"{100 * a['within_20']:.0f}%", f"{a['span_median']:.1f}", f"{a['span_mean']:.1f}", f"{a['span_max']:.0f}",
                         f"{100 * a['verdict_change_rate']:.1f}%"])
        by_tier = {}
        for t, gt in g.assign(tier=g.firm_id.map(tier_of)).groupby("tier"):
            if gt.firm_id.nunique() >= 2:
                rt = M.icc_oneway(gt.prob_fail, gt.firm_id, n_boot=0, seed=seed)
                by_tier[t] = rt["icc1"]
                tier_rows.append([f"{temp:g}", t, gt.firm_id.nunique(), f3(rt["icc1"]), f"{rt['sem']:.2f}"])
        js[str(temp)] = dict(icc=r, agreement=a, icc1_by_tier=by_tier)
    R.table(["Temp.", "Firms", "Ratings", "k₀", "ICC(1)", "F-CI", "boot-CI", "ICC(1,k)", "F-CI", "boot-CI", "SEM", "SDC₉₅"], rows)
    R.text("Rep-to-rep agreement (share of firms):")
    R.table(["Temp.", "Firms", "Identical", "≤5 pts", "≤10 pts", "≤20 pts", "Span median", "Span mean", "Span max",
             f"Verdict change (≥{THRESHOLD:.0f})"], agr_rows)
    R.text("ICC(1) by expert tier:")
    R.table(["Temp.", "Tier", "Firms", "ICC(1)", "SEM"], tier_rows)
    R.json["table_VC_reliability"] = js


def table_operating(df, R, pop_prev, title, key):
    R.section(title, f"Flag = prob_fail ≥ {THRESHOLD:.0f}. Capture = sensitivity; burden = share of registry-normal firms flagged. "
              f"PPV/NPV re-weighted to the population Below-A prevalence {100 * pop_prev:.2f}% (Bayes, from sensitivity/specificity).")
    rows, js = [], {}
    for col in LLM_COLS:
        if col not in df:
            continue
        c = M.confusion_at_threshold(df.failure, df[col], THRESHOLD)
        p = M.ppv_npv_at_prevalence(c["capture"], c["specificity"], pop_prev)
        rows.append([LABEL[col], c["n"], c["TP"], c["FP"], c["FN"], c["TN"], f"{100 * c['capture']:.1f}%", f"{100 * c['burden']:.1f}%",
                     f"{100 * c['ppv_sample']:.1f}%", f"{100 * p['ppv']:.1f}%", f"{100 * p['npv']:.2f}%", f"{100 * p['flag_rate']:.1f}%"])
        js[col] = dict(confusion=c, at_population_prevalence=p)
    R.table(["Model", "n", "TP", "FP", "FN", "TN", "Capture", "Burden", "PPV (sample)", "PPV (pop.)", "NPV (pop.)", "Flag rate (pop.)"], rows)
    R.json[key] = js


def table_sector(df, R, key):
    R.section("Table SEC — Sector-stratified AUROC (Below-A frozen set)",
              "Pooled within-ksic2 AUROC = P(score_fail > score_normal | same 2-digit division), pair-weighted over divisions with "
              "both classes present; compared with the unstratified AUROC on the rows used.")
    rows, js = [], {}
    for col in ["grade13", "score_tfidf_oof", "score_sonnet", "score_gpt55", "score_qwen35"]:
        if col not in df:
            continue
        sub = df.dropna(subset=[col])
        r = M.stratified_auroc(sub.failure, sub[col], sub.ksic2)
        rows.append([LABEL[col], r["n_rows_used"], r["n_strata_used"], f3(r["pooled_unstratified_auroc"]), f3(r["pooled_auroc"]),
                     f"{r['pooled_auroc'] - r['pooled_unstratified_auroc']:+.3f}"])
        js[col] = {k: v for k, v in r.items() if k != "per_stratum"}
        js[col]["n_per_stratum_median"] = float(np.median([s["n"] for s in r["per_stratum"]])) if r["per_stratum"] else None
    R.table(["Predictor", "Rows used", "Divisions", "Unstratified AUROC", "Within-division pooled AUROC", "Δ"], rows)
    # sector-mean-rate prior as a reference (leave-one-out)
    g = df.groupby("ksic2").failure.agg(["sum", "size"])
    j = df.join(g, on="ksic2")
    loo = (j["sum"] - j.failure) / (j["size"] - 1).replace(0, np.nan)
    m = loo.notna()
    a = M.auroc(df.failure[m], loo[m])
    R.text(f"Reference: leave-one-out sector failure-rate prior (no text) — AUROC {a:.3f} on {int(m.sum())} rows.")
    js["sector_rate_prior_loo"] = dict(auroc=a, n=int(m.sum()))
    R.json[key] = js


def table_triage(df, R, pop_prev, seed, key, rule=None):
    rule = rule or dict(accept_ratio=0.5, reject_lift=1.5, min_band_n=20)
    R.section("Table TRI — Cross-validated two-sided triage (Sonnet score, Below-A frozen set)",
              f"5-fold stratified CV (seed {seed}): on each training fold, clear-accept = largest cut-off whose band failure rate is "
              f"≤ {rule['accept_ratio']}× the training prevalence; clear-reject = smallest cut-off whose band failure rate is "
              f"≥ {rule['reject_lift']}× the prevalence (bands ≥ {rule['min_band_n']} firms). Routing is evaluated out-of-fold. "
              f"False accepts = failures routed to accept; false rejects = registry-normal firms routed to reject.")
    js = {}
    rows = []
    for col in ["score_sonnet", "score_tfidf_oof", "grade13"]:
        if col not in df:
            continue
        sub = df.dropna(subset=[col])
        t = M.triage_cv(sub.failure, sub[col], n_splits=5, seed=seed, **rule)
        rw = M.triage_reweight(t, pop_prev)
        b = t["bands"]
        rows.append([LABEL[col], t["n"],
                     f"{100 * b['accept']['share']:.1f}% ({b['accept']['n']})", f"{100 * b['review']['share']:.1f}% ({b['review']['n']})",
                     f"{100 * b['reject']['share']:.1f}% ({b['reject']['n']})",
                     f"{t['false_accepts']} / {t['n_pos']}", f"{t['false_rejects']} / {t['n_neg']}",
                     f"{100 * b['accept']['fail_rate']:.1f}%" if b["accept"]["n"] else "—",
                     f"{100 * b['reject']['fail_rate']:.1f}%" if b["reject"]["n"] else "—",
                     f"{100 * rw['accept']['share']:.1f}% / {100 * rw['review']['share']:.1f}% / {100 * rw['reject']['share']:.1f}%",
                     f"{100 * rw['accept']['fail_rate']:.2f}%" if rw["accept"]["share"] > 0 else "—",
                     f"{100 * rw['reject']['fail_rate']:.1f}%" if rw["reject"]["share"] > 0 else "—"])
        js[col] = dict(oof=t["bands"], false_accepts=t["false_accepts"], false_rejects=t["false_rejects"], n=t["n"],
                       n_pos=t["n_pos"], n_neg=t["n_neg"], folds=t["folds"], reweighted=rw, rule=t["rule"])
    R.table(["Predictor", "n", "Accept (OOF)", "Review (OOF)", "Reject (OOF)", "False accepts", "False rejects",
             "Fail-rate accept", "Fail-rate reject", "Routing at pop. prevalence (acc/rev/rej)",
             "Accept-band fail-rate (pop.)", "Reject-band fail-rate (pop.)"], rows)
    R.json[key] = js


def table_screener(scr, R, n_boot, seed):
    R.section("Table V-3 — Two-sided coarse screen (A-vs-rest, C-vs-rest)",
              f"n = {len(scr)} ({(scr.expert_tier == 'A').sum()} A / {(scr.expert_tier == 'B').sum()} B / {(scr.expert_tier == 'C').sum()} C). "
              "Quality score is higher-is-better; C-vs-rest uses the negated score. Bootstrap CIs as above.")
    rows, js = [], {}
    yA = (scr.expert_tier == "A").astype(int)
    yC = (scr.expert_tier == "C").astype(int)
    for col, label in [("quality_score_bare", "Bare (label-free) LLM"), ("quality_score_anchor", "Anchor-primed LLM")]:
        if col not in scr:
            continue
        a = M.bootstrap_auroc_ci(yA, scr[col], n_boot=n_boot, seed=seed)
        c = M.bootstrap_auroc_ci(yC, -scr[col], n_boot=n_boot, seed=seed)
        rows.append([label, a["n"], f3(a["auroc"]), ci(a["lo"], a["hi"]), f3(c["auroc"]), ci(c["lo"], c["hi"])])
        js[col] = dict(A_vs_rest=a, C_vs_rest=c)
    R.table(["Method", "n", "A-vs-rest AUROC", "95% CI", "C-vs-rest AUROC", "95% CI"], rows)
    R.json["table_V3"] = js


# ----------------------------------------------------------------------------- figures
def figures(fz_below, rel, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception as e:  # noqa: BLE001
        print(f"[figures skipped: {e}]")
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for col in ["grade13", "score_tfidf_oof", "score_sonnet", "desc_len_chars"]:
        sub = fz_below.dropna(subset=[col])
        fpr, tpr, _ = roc_curve(sub.failure, sub[col])
        ax.plot(fpr, tpr, label=f"{LABEL[col]} ({M.auroc(sub.failure, sub[col]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("registry-normal flagged (burden)")
    ax.set_ylabel("verified failures captured")
    ax.set_title("Below-A frozen set — ROC")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_roc_below_A.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, temp in zip(axes, sorted(rel.temperature.unique(), reverse=True)):
        g = rel[rel.temperature == temp].groupby("firm_id").prob_fail
        span = g.max() - g.min()
        ax.hist(span, bins=np.arange(0, span.max() + 2) - 0.5, color="#4C72B0")
        ax.set_title(f"temperature {temp:g}: span across {int(g.size().median())} re-scorings")
        ax.set_xlabel("max − min prob_fail (points)")
        ax.set_ylabel("firms")
    fig.tight_layout()
    fig.savefig(out / "fig_reliability_spans.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="directory with firms.csv, reliability.csv, screener.csv")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <data>/output)")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for AUROC / ICC CIs")
    ap.add_argument("--tost-boot", type=int, default=4000, help="paired-bootstrap resamples for TOST")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (paper: 0)")
    ap.add_argument("--cv-seed", type=int, default=42, help="fold seed for the triage CV (paper: 42)")
    ap.add_argument("--triage-accept-ratio", type=float, default=0.5,
                    help="clear-accept band: max failure rate as a multiple of the training prevalence")
    ap.add_argument("--triage-reject-lift", type=float, default=1.5,
                    help="clear-reject band: min failure rate as a multiple of the training prevalence")
    ap.add_argument("--triage-min-band-n", type=int, default=20)
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out = args.out or (args.data / "output")
    out.mkdir(parents=True, exist_ok=True)

    firms = pd.read_csv(args.data / "firms.csv", dtype={"ksic2": str, "firm_id": str})
    rel = pd.read_csv(args.data / "reliability.csv", dtype={"firm_id": str})
    scr = pd.read_csv(args.data / "screener.csv", dtype={"firm_id": str})
    required = {"firm_id", "tier", "grade13", "ksic2", "desc_len_chars", "failure", "score_sonnet", "in_frozen_set"}
    missing = required - set(firms.columns)
    if missing:
        sys.exit(f"firms.csv is missing required columns: {sorted(missing)}")

    R = Report()
    R.md += ["# Reproduction tables", "",
             f"Data: `{args.data}` — firms.csv {len(firms):,} rows, reliability.csv {len(rel):,} ratings, screener.csv {len(scr):,} rows. "
             f"Bootstrap {args.n_boot} (seed {args.seed}); TOST {args.tost_boot}; CV seed {args.cv_seed}.", ""]

    comp = table_composition(firms, R)
    pop_prev = comp["below_A"]["rate"]

    fz = firms[firms.in_frozen_set == 1].copy()
    fz_below = fz[fz.tier != "A"].copy()
    fz_b = fz[fz.tier == "B"].copy()

    table_plateau(fz_below, R, "Table V-1 — Method-invariant plateau on the Below-A frozen set", "table_V1_below_A", args.n_boot, args.seed)
    table_plateau(fz_b, R, "Table V-1b — Within the B-tier (frozen set)", "table_V1b_B_tier", args.n_boot, args.seed)
    pairs = [("score_sonnet", "grade13"), ("score_sonnet", "score_tfidf_oof"), ("grade13", "score_tfidf_oof"),
             ("score_sonnet", "score_gpt55"), ("score_sonnet", "score_qwen35"), ("score_sonnet", "score_deepseek"),
             ("score_sonnet", "score_llama"), ("score_sonnet", "desc_len_chars"), ("grade13", "desc_len_chars")]
    table_pairwise(fz_below, R, "Table V-1c — Pairwise DeLong tests and TOST equivalence (Below-A frozen set)", "table_V1c_pairwise_below_A",
                   pairs, args.tost_boot, args.seed)
    table_pairwise(fz_b, R, "Table V-1d — Pairwise DeLong tests and TOST equivalence (B-tier frozen set)", "table_V1d_pairwise_B_tier",
                   pairs[:3], args.tost_boot, args.seed)

    # population-level check if scores exist outside the frozen set (true for synthetic data)
    pop_scored = firms[(firms.in_frozen_set == 0) & firms.score_sonnet.notna()]
    if len(pop_scored) > 500:
        R.section("Supplement — AUROC on the full labeled population (scores available outside the frozen set)",
                  "Present only when the extract carries LLM scores for firms outside the frozen set (synthetic data does).")
        rows, js = [], {}
        for name, sub in [("Full corpus", firms), ("Below-A", firms[firms.tier != "A"]), ("B-tier", firms[firms.tier == "B"]),
                          ("A-tier", firms[firms.tier == "A"]), ("C-tier", firms[firms.tier == "C"])]:
            for col in ["grade13", "score_sonnet", "desc_len_chars"]:
                b = M.bootstrap_auroc_ci(sub.failure, sub[col], n_boot=min(args.n_boot, 500), seed=args.seed)
                rows.append([name, LABEL[col], b["n"], b["n_pos"], f3(b["auroc"]), ci(b["lo"], b["hi"])])
                js[f"{name}__{col}"] = b
        R.table(["Population", "Predictor", "n", "Failures", "AUROC", "95% CI (500 boots)"], rows)
        R.json["supplement_population"] = js

    table_style(fz, R)
    table_reliability(rel, firms, R, args.n_boot, args.seed)
    table_operating(fz_below, R, pop_prev, "Table OP — Operating points on the Below-A frozen set", "table_OP_below_A")
    table_sector(fz_below, R, "table_SEC_below_A")
    table_triage(fz_below, R, pop_prev, args.cv_seed, "table_TRI_below_A",
                 rule=dict(accept_ratio=args.triage_accept_ratio, reject_lift=args.triage_reject_lift,
                           min_band_n=args.triage_min_band_n))
    table_screener(scr, R, args.n_boot, args.seed)

    if not args.no_figs:
        figures(fz_below, rel, out)

    R.json["_meta"] = dict(data=str(args.data), n_boot=args.n_boot, tost_boot=args.tost_boot, seed=args.seed,
                           cv_seed=args.cv_seed, runtime_s=time.time() - t0)
    (out / "tables.md").write_text("\n".join(R.md))
    (out / "results.json").write_text(json.dumps(R.json, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print("\n".join(R.md))
    print(f"\nwrote {out / 'tables.md'} and {out / 'results.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
