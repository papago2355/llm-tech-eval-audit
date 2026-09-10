"""build_extract.py — derive the de-identified extract (SCHEMA.md) from the proprietary source files.

FOR THE DATA HOLDERS ONLY. This script documents exactly how firms.csv / reliability.csv / screener.csv are
produced from the source workbook and the run logs, so that the extract handed to third parties is auditable.
It reads the proprietary files, which are NOT part of this repository, and writes the extract to a git-ignored
directory (default data/private/extract). The extract contains no names, no description text, no free text.

Inputs (defaults assume the layout of the authors' analysis repository, with this package in <root>/release):
  --source      <root>/original_tech_evaluation_xlsx.xlsx (sheet "data")
  --frozen      <root>/experiments/24_backbone_failures/test_set.csv          (frozen 967-row set)
  --scores-dir  <root>/experiments/24_backbone_failures/outputs                (per-model CSVs: row_id, prob_fail)
  --rel-jsonl   <root>/experiments/25_reliability_retest/reliability.jsonl     (firm_id, temp, rep, prob_fail, error)
  --rel-sample  <root>/experiments/25_reliability_retest/reliability_sample.csv (firm_id -> description)
  --coarse-dir  <root>/experiments/26_apprentice_evaluator                     (coarse.jsonl, coarse_bare.jsonl, coarse_test.csv)

Joins between files are made on the exact description text. Identical texts occur for several firms, so each
external row is assigned to one not-yet-used labeled row, preferring a candidate with the same grade and status
(TextMatcher). Firm ids are re-keyed with a seeded random permutation and every file is written in firm_id order, so neither the key nor the row position can be aligned with the
source row order. Report lines print only counts.

Usage:
  python tools/build_extract.py --out data/private/extract [--seed 20260906]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # <root>/release/tools -> <root>

GRADE13 = ["AAA", "AA", "A+", "A", "BBB+", "BBB", "BB+", "BB", "B+", "B", "CCC", "CC", "C"]
RANK = {g: i for i, g in enumerate(GRADE13)}
COL_DESC, COL_GRADE, COL_STATUS, COL_SECTOR = "기술개요", "최종등급", "업체상태", "업종"
LABEL_FAIL, LABEL_NORMAL = "사고", "정상"
MODEL_FILES = {"score_sonnet": "sonnet-4.6.csv", "score_gpt55": "gpt5.5.csv", "score_qwen35": "qwen3.5.csv",
               "score_deepseek": "deepseek-v4.csv", "score_llama": "llama4-mav.csv"}
SONNET_ID = "anthropic/claude-sonnet-4.6"


def tier_of(g: str) -> str:
    return "A" if g in {"AAA", "AA", "A+", "A"} else ("C" if str(g).startswith("C") else "B")


def auroc(y, s) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tfidf_oof(texts: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Exact TABLE 3 TF-IDF protocol (experiments/24_backbone_failures/tfidf_baseline.py, char_wb 2-4)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, texts, y, cv=skf, method="predict_proba")[:, 1]


class TextMatcher:
    """Deterministic 1:1 assignment of external rows to labeled-corpus rows by exact description text.

    Identical texts occur for several firms; a candidate with the same grade and status is preferred and each
    labeled row is used at most once (in the external file's row order). Use a fresh matcher per external file.
    """

    def __init__(self, lab: pd.DataFrame):
        self.lab = lab
        self.pool: dict[str, list[int]] = {}
        for i, t in zip(lab.index, lab[COL_DESC].astype(str)):
            self.pool.setdefault(t, []).append(i)

    def match(self, text, grade=None, status=None):
        cands = self.pool.get(str(text), [])
        if not cands:
            return np.nan
        pick = None
        for i in cands:
            ok = (grade is None or pd.isna(grade) or self.lab.at[i, COL_GRADE] == grade) and \
                 (status is None or pd.isna(status) or self.lab.at[i, COL_STATUS] == status)
            if ok:
                pick = i
                break
        if pick is None:
            pick = cands[0]
        cands.remove(pick)
        return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ROOT / "original_tech_evaluation_xlsx.xlsx")
    ap.add_argument("--sheet", default="data")
    ap.add_argument("--frozen", type=Path, default=ROOT / "experiments/24_backbone_failures/test_set.csv")
    ap.add_argument("--scores-dir", type=Path, default=ROOT / "experiments/24_backbone_failures/outputs")
    ap.add_argument("--rel-jsonl", type=Path, default=ROOT / "experiments/25_reliability_retest/reliability.jsonl")
    ap.add_argument("--rel-sample", type=Path, default=ROOT / "experiments/25_reliability_retest/reliability_sample.csv")
    ap.add_argument("--coarse-dir", type=Path, default=ROOT / "experiments/26_apprentice_evaluator")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data/private/extract")
    ap.add_argument("--seed", type=int, default=20260906, help="seed of the firm-id re-keying permutation")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- labeled corpus
    d = pd.read_excel(args.source, sheet_name=args.sheet)
    d.columns = [c.replace("\n", "").strip() for c in d.columns]
    lab = d[d[COL_STATUS].isin([LABEL_FAIL, LABEL_NORMAL])].copy().reset_index(drop=True)
    lab["failure"] = (lab[COL_STATUS] == LABEL_FAIL).astype(int)
    lab["grade13"] = lab[COL_GRADE].map(RANK)
    lab["tier"] = lab[COL_GRADE].map(tier_of)
    lab["ksic2"] = lab[COL_SECTOR].astype(str).str.extract(r"\((\d+)\)")[0].str[:2]
    lab["desc_len_chars"] = lab[COL_DESC].fillna("").astype(str).str.len().astype(int)
    print(f"labeled firms: {len(lab)} | failures: {int(lab.failure.sum())} | grade missing: {int(lab.grade13.isna().sum())} | "
          f"ksic2 missing: {int(lab.ksic2.isna().sum())} | rows sharing a description text: {int(lab[COL_DESC].duplicated(keep=False).sum())}")
    for t in ("A", "B", "C"):
        g = lab[lab.tier == t]
        print(f"  tier {t}: {len(g)} firms, {int(g.failure.sum())} failures")

    rng = np.random.default_rng(args.seed)
    lab["firm_id"] = [f"F{i:05d}" for i in rng.permutation(len(lab))]

    # ---------------------------------------------------------------- frozen set + model scores
    fz = pd.read_csv(args.frozen)
    tm = TextMatcher(lab)
    fz["idx"] = [tm.match(t, g, s) for t, g, s in zip(fz[COL_DESC], fz.get(COL_GRADE), fz.get(COL_STATUS))]
    print(f"frozen rows: {len(fz)} | matched to labeled corpus by text: {int(fz.idx.notna().sum())}")
    fz = fz.dropna(subset=["idx"])
    fz["idx"] = fz.idx.astype(int)
    lab["in_frozen_set"] = 0
    lab.loc[fz.idx, "in_frozen_set"] = 1
    for col in MODEL_FILES:
        lab[col] = np.nan
    for col, fn in MODEL_FILES.items():
        p = args.scores_dir / fn
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        sc = pd.read_csv(p)[["row_id", "prob_fail"]]
        m = fz[["row_id", "idx"]].merge(sc, on="row_id", how="left")
        lab.loc[m.idx.values, col] = m.prob_fail.values
        print(f"  {col}: {int(m.prob_fail.notna().sum())} scores")

    # TF-IDF out-of-fold within the Below-A frozen set (exact TABLE 3 protocol)
    lab["score_tfidf_oof"] = np.nan
    ba = fz[(lab.loc[fz.idx, "tier"].values != "A") & fz[COL_DESC].notna().values].copy()
    y = lab.loc[ba.idx, "failure"].values
    oof = tfidf_oof(ba[COL_DESC].astype(str).values, y, seed=42)
    lab.loc[ba.idx.values, "score_tfidf_oof"] = oof
    print(f"  score_tfidf_oof: {len(oof)} rows (Below-A frozen), AUROC {auroc(y, oof):.3f}")

    cols = ["firm_id", "tier", "grade13", "ksic2", "desc_len_chars", "failure", *MODEL_FILES, "score_tfidf_oof", "in_frozen_set"]
    firms = lab[cols].copy()
    firms["grade13"] = firms.grade13.astype("Int64")
    firms.sort_values("firm_id", kind="stable").to_csv(args.out / "firms.csv", index=False)

    # ---------------------------------------------------------------- reliability
    rs = pd.read_csv(args.rel_sample)
    tm = TextMatcher(lab)
    status = rs["fail"].map({1: LABEL_FAIL, 0: LABEL_NORMAL}) if "fail" in rs else [None] * len(rs)
    rs["idx"] = [tm.match(t, None, s) for t, s in zip(rs[COL_DESC], status)]
    rel = pd.DataFrame([json.loads(l) for l in args.rel_jsonl.read_text().splitlines() if l.strip()])
    rel = rel.merge(rs[["firm_id", "idx"]], on="firm_id", how="left")
    print(f"reliability calls: {len(rel)} | firms: {rel.firm_id.nunique()} | matched: {int(rel.idx.notna().sum())} | "
          f"errors: {int(rel['error'].notna().sum())}")
    rel = rel.dropna(subset=["idx"])
    out_rel = pd.DataFrame({"firm_id": lab.loc[rel.idx.astype(int), "firm_id"].values,
                            "temperature": rel["temp"].astype(float).values, "rep": rel["rep"].astype(int).values,
                            "prob_fail": pd.to_numeric(rel["prob_fail"], errors="coerce").values})
    out_rel.sort_values("firm_id", kind="stable").to_csv(args.out / "reliability.csv", index=False)

    # ---------------------------------------------------------------- screener (Sonnet rows only)
    ct = pd.read_csv(args.coarse_dir / "coarse_test.csv")
    tm = TextMatcher(lab)
    ct["idx"] = [tm.match(t, g, s) for t, g, s in zip(ct[COL_DESC], ct.get(COL_GRADE), ct.get(COL_STATUS))]
    unl = ct.idx.isna()  # unlabeled firms keep an independent key
    ct["firm_id"] = [f"U{i:05d}" if u else lab.at[int(ix), "firm_id"] for i, (u, ix) in enumerate(zip(unl, ct.idx.fillna(-1)))]
    scr = ct[["row_id", "firm_id", "tier"]].rename(columns={"tier": "expert_tier"})
    for col, fn in [("quality_score_anchor", "coarse.jsonl"), ("quality_score_bare", "coarse_bare.jsonl")]:
        j = pd.DataFrame([json.loads(l) for l in (args.coarse_dir / fn).read_text().splitlines() if l.strip()])
        j = j[j.model == SONNET_ID][["row_id", "pred_score"]].rename(columns={"pred_score": col})
        j[col] = pd.to_numeric(j[col], errors="coerce")
        scr = scr.merge(j, on="row_id", how="left")
    print(f"screener rows: {len(scr)} | labeled-corpus matches: {int((~unl).sum())} | "
          f"anchor scores: {int(scr.quality_score_anchor.notna().sum())} | bare scores: {int(scr.quality_score_bare.notna().sum())}")
    scr.drop(columns=["row_id"])[["firm_id", "expert_tier", "quality_score_bare", "quality_score_anchor"]].sort_values("firm_id", kind="stable").to_csv(args.out / "screener.csv", index=False)
    print(f"wrote {args.out}/firms.csv, reliability.csv, screener.csv")


if __name__ == "__main__":
    main()
