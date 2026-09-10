"""Unit tests for analysis/metrics.py.

Run:  python -m pytest release/analysis/test_metrics.py -q
  or: python release/analysis/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import metrics as M  # noqa: E402


def _data(seed=0, n=300, ties=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(float)
    s1 = rng.normal(0.4 * y, 1)
    s2 = 0.6 * s1 + rng.normal(0, 1, n)
    if ties:  # LLM scores are coarse: force many ties
        s1 = np.round(s1 * 2) / 2
        s2 = np.round(s2 * 2) / 2
    return y, s1, s2


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    for seed in range(5):
        y, s1, _ = _data(seed)
        assert abs(M.auroc(y, s1) - roc_auc_score(y, s1)) < 1e-12


def test_delong_fast_equals_brute_force():
    """Sun & Xu (2014) midrank algorithm must reproduce the O(mn) structural components."""
    for seed in range(5):
        y, s1, s2 = _data(seed, n=250)
        aucs, S = M.delong_components(y, np.vstack([s1, s2]))
        for r, s in enumerate((s1, s2)):
            auc_b, var_b, v10_b, v01_b = M._brute_force_delong_components(y, s)
            assert abs(aucs[r] - auc_b) < 1e-12
            assert abs(S[r, r] - var_b) < 1e-12
        # covariance between the two predictors, brute force
        _, _, v10_1, v01_1 = M._brute_force_delong_components(y, s1)
        _, _, v10_2, v01_2 = M._brute_force_delong_components(y, s2)
        m, n = int(y.sum()), int(len(y) - y.sum())
        cov_b = np.cov(v10_1, v10_2)[0, 1] / m + np.cov(v01_1, v01_2)[0, 1] / n
        assert abs(S[0, 1] - cov_b) < 1e-12


def test_delong_test_identical_scores_is_null():
    y, s1, _ = _data(1)
    r = M.delong_test(y, s1, s1.copy())
    assert r["diff"] == 0.0 and r["se"] == 0.0


def test_delong_p_is_calibrated_under_null():
    """Two independent noise predictors: DeLong p should be ~Uniform(0,1) (reject ~5% at 0.05)."""
    rng = np.random.default_rng(7)
    rej = 0
    B = 400
    for _ in range(B):
        y = (rng.random(200) < 0.4).astype(float)
        s1, s2 = rng.normal(size=200), rng.normal(size=200)
        rej += M.delong_test(y, s1, s2)["p"] < 0.05
    assert 0.02 < rej / B < 0.10


def test_bootstrap_ci_contains_point():
    y, s1, _ = _data(2)
    r = M.bootstrap_auroc_ci(y, s1, n_boot=300, seed=0)
    assert r["lo"] <= r["auroc"] <= r["hi"]
    assert r["n_boot_used"] == 300


def test_tost_symmetry_and_equivalence():
    y, s1, s2 = _data(3, n=2000)
    r12 = M.tost_equivalence(y, s1, s2, margin=0.5, n_boot=300)
    r21 = M.tost_equivalence(y, s2, s1, margin=0.5, n_boot=300)
    assert abs(r12["diff"] + r21["diff"]) < 1e-12
    assert r12["equivalent_boot"] and r12["p_tost"] < 0.05  # wide margin -> equivalent


def test_icc_balanced_against_textbook_formulas():
    """Balanced design: ICC(1) = (MSB-MSW)/(MSB+(k-1)MSW) and F-CI reproduce closed forms."""
    rng = np.random.default_rng(0)
    a, k = 40, 5
    firm = rng.normal(0, 15, a)
    vals = (firm[:, None] + rng.normal(0, 2, (a, k))).ravel()
    groups = np.repeat(np.arange(a), k)
    r = M.icc_oneway(vals, groups, n_boot=200)
    X = vals.reshape(a, k)
    msb = k * X.mean(1).var(ddof=1)
    msw = X.var(axis=1, ddof=1).mean()
    assert abs(r["MSB"] - msb) < 1e-9 and abs(r["MSW"] - msw) < 1e-9
    assert abs(r["icc1"] - (msb - msw) / (msb + (k - 1) * msw)) < 1e-12
    assert abs(r["icc1k"] - (msb - msw) / msb) < 1e-12
    assert r["icc1_f_ci"][0] < r["icc1"] < r["icc1_f_ci"][1]
    assert r["icc1_boot_ci"][0] < r["icc1"] < r["icc1_boot_ci"][1]
    assert 0.97 < r["icc1"] < 0.995  # 15^2/(15^2+2^2) = 0.983


def test_icc_unbalanced_k0():
    vals = np.array([10, 11, 10, 20, 21, 30, 31, 29, 30.0])
    groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    r = M.icc_oneway(vals, groups, n_boot=50)
    n, a = 9, 3
    k0 = (n - (9 + 4 + 16) / n) / (a - 1)
    assert abs(r["k0"] - k0) < 1e-12
    assert r["icc1"] > 0.95


def test_agreement_and_verdict_change():
    vals = np.array([40, 40, 40, 48, 56, 50, 60, 61, 62])
    groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    r = M.agreement_proportions(vals, groups, threshold=50)
    assert r["identical"] == 1 / 3 and r["verdict_change_rate"] == 1 / 3
    assert r["within_5"] == 2 / 3 and r["within_10"] == 1.0


def test_confusion_and_reweighting():
    y = np.array([1, 1, 0, 0, 0, 1])
    s = np.array([60, 40, 55, 20, 30, 50])
    c = M.confusion_at_threshold(y, s, 50)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (2, 1, 1, 2)
    p = M.ppv_npv_at_prevalence(c["capture"], c["specificity"], 0.5)
    assert abs(p["ppv"] - c["ppv_sample"]) < 1e-12  # sample prevalence is 0.5 here


def test_stratified_auroc_removes_between_stratum_shift():
    rng = np.random.default_rng(3)
    strata = np.repeat(["a", "b"], 500)
    y = (rng.random(1000) < 0.3).astype(float)
    s = rng.normal(0, 1, 1000) + np.where(strata == "b", 5.0, 0.0)  # pure between-stratum shift
    r = M.stratified_auroc(y, s, strata)
    assert abs(r["pooled_auroc"] - 0.5) < 0.05
    assert r["n_strata_used"] == 2


def test_pu_implied_auroc_identity():
    """Hiding a random fraction of positives among the negatives deflates AUROC by (1-a)*A + a/2, and the inverse recovers A."""
    rng = np.random.default_rng(3)
    n = 60000
    y = (rng.random(n) < 0.3).astype(int)
    s = y * 1.0 + rng.normal(0, 1.5, n)
    a_true = M.auroc(y, s)
    alpha = 0.10
    neg_n = int((y == 0).sum())
    k = int(round(alpha * neg_n / (1 - alpha)))          # hidden positives so that k / (neg_n + k) = alpha
    hide = rng.choice(np.flatnonzero(y == 1), size=k, replace=False)
    y_obs = y.copy()
    y_obs[hide] = 0
    a_obs = M.auroc(y_obs, s)
    assert abs(a_obs - ((1 - alpha) * a_true + alpha / 2)) < 0.01, (a_obs, a_true)
    assert abs(M.pu_implied_auroc(a_obs, alpha) - a_true) < 0.01
    assert M.pu_implied_auroc(0.7, 0.0) == 0.7


def test_purity_thresholds_rule():
    """Lowest admissible accept cut-off / highest admissible reject cut-off; overlap routes to human review."""
    scores = np.arange(1, 11, dtype=float)
    tiers = np.array(["C"] * 5 + ["A"] * 5)
    t_acc, t_rej = M.purity_thresholds(scores, tiers, purity=0.80, min_n=5)
    assert t_acc == 5.0, t_acc          # scores >= 5: 6 firms, 5 A (0.83); scores >= 4: 7 firms, 5 A (0.71)
    assert t_rej == 6.0, t_rej          # scores <= 6: 6 firms, 5 C (0.83); scores <= 7: 7 firms, 5 C (0.71)
    bins = M.purity_route(scores, t_acc, t_rej)
    assert list(bins[:4]) == ["auto_reject"] * 4 and list(bins[6:]) == ["auto_accept"] * 4
    assert list(bins[4:6]) == ["human_review", "human_review"]   # 5 and 6 satisfy both rules
    assert M.purity_thresholds(scores, tiers, purity=1.0, min_n=6) == (None, None)
    cv = M.purity_triage_cv(scores, tiers, purity=0.8, n_repeats=2, n_splits=2, min_n=2)
    assert abs(sum(cv["share"].values()) - 1.0) < 1e-12 and cv["folds"] == 4


def test_triage_cv_routes_everyone_once():
    y, s1, _ = _data(5, n=800)
    r = M.triage_cv(y, s1 * 10 + 50, n_splits=5, seed=1)
    tot = sum(b["n"] for b in r["bands"].values())
    assert tot == r["n"]
    rw = M.triage_reweight(r, 0.059)
    assert abs(sum(rw[b]["share"] for b in ("accept", "review", "reject")) - 1) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:  # noqa: PERF203
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
