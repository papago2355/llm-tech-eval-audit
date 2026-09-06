# Data schema for the analysis scripts

`analysis/reproduce.py --data <dir>` expects three UTF-8 CSV files in `<dir>`. The synthetic
generator writes exactly this schema; the de-identified real extract (available from the authors on
reasonable request, subject to the data owner's approval) uses the same schema, so the identical
scripts run on both. No file contains a firm name, a description text, or any free text.

## firms.csv — one row per labeled firm (population; real extract: 9,926 rows)

| column | type | definition |
|---|---|---|
| `firm_id` | string | opaque, de-identified key (synthetic: `S00000`…; real extract: a random re-keying, not the source row number) |
| `tier` | `A` / `B` / `C` | expert grade tier. A = {AAA, AA, A+, A}; C = grade starts with "C" (CCC, CC, C); B otherwise |
| `grade13` | int 0–12 | ordinal index of the 13-level expert grade in `AAA, AA, A+, A, BBB+, BBB, BB+, BB, B+, B, CCC, CC, C` (0 = AAA … 12 = C; **higher = riskier**, so it is used directly as a risk score) |
| `ksic2` | string (2 chars) | 2-digit KSIC industry division = first two digits of the 5-digit code in the industry field. Keep as a string (leading zeros). Synthetic data uses 100 synthetic codes `00`–`99` |
| `desc_len_chars` | int | character count of the technology description (the only text the LLMs saw). The "usable" filter of the paper is 80 ≤ len ≤ 2000 |
| `failure` | 0 / 1 | 1 = verified failure event by end-June 2023 (source status "사고"); 0 = registry-normal ("정상"). Unlabeled firms are excluded from the file |
| `score_sonnet` | float 0–100 or empty | `prob_fail` from Claude Sonnet 4.6 with prompt `blind_direct_v1` at temperature 0, seed 42 (paper's cached run). Empty where the firm was not scored |
| `score_gpt55` | float or empty | same protocol, `openai/gpt-5.5` |
| `score_qwen35` | float or empty | same protocol, `qwen/qwen3.5-27b` |
| `score_deepseek` | float or empty | same protocol, `deepseek/deepseek-v4-flash` (2 unparsable outputs in the real run → empty) |
| `score_llama` | float or empty | same protocol, `meta-llama/llama-4-maverick` |
| `score_tfidf_oof` | float 0–1 or empty | out-of-fold probability of the TF-IDF baseline (char_wb 2–4 n-grams, min_df 2, 50k features; balanced logistic regression C=1; StratifiedKFold(5, shuffle, random_state=42)) fit **within the Below-A frozen set** — defined only on those 941 rows |
| `in_frozen_set` | 0 / 1 | 1 for the 967 rows of the frozen evaluation set: all Below-A failures with a usable description (377), 564 sampled registry-normal firms (550 B-tier + 14 C-tier), and 26 A-tier failures |

Derived populations used by `reproduce.py`:

* **Below-A population** = `tier != "A"` (Table I: 8,460 firms, 499 failures → prevalence 5.9%, used for PPV re-weighting).
* **Below-A frozen set** = `in_frozen_set == 1 & tier != "A"` (941 rows = 377 failures + 564 normals): Tables V-1, V-1c, OP, SEC, TRI.
* **B-tier frozen set** = `in_frozen_set == 1 & tier == "B"` (878 rows = 328 + 550): Tables V-1b, V-1d.
* **Frozen set** (967 rows): Table V-2.

In the real extract the five `score_*` columns are populated only for `in_frozen_set == 1` rows (plus the
reliability-sample firms); the synthetic generator fills them for every firm, which additionally enables
the population-level supplement table.

## reliability.csv — one row per re-scoring call (real extract: 1,668 rows = 139 firms × 12 calls)

| column | type | definition |
|---|---|---|
| `firm_id` | string | joins to `firms.csv` |
| `temperature` | float | `0.7` (stress test, no seed, 10 reps) or `0.0` (deployment setting, seed 42 + rep, 2 reps) |
| `rep` | int | repetition index within (firm, temperature) |
| `prob_fail` | float 0–100 or empty | parsed score; empty = unparsable output (dropped by the analysis) |

Sampling: 25 firms per tier × failure cell from firms with a usable description (the C-tier registry-normal
cell has only 14) → 139 firms. Model `anthropic/claude-sonnet-4.6`, prompt `blind_direct_v1`, every call a
fresh API hit on byte-identical text.

## screener.csv — one row per firm of the coarse-screen test set (real extract: 230 rows)

| column | type | definition |
|---|---|---|
| `firm_id` | string | joins to `firms.csv` (or an independent key if the firm is unlabeled) |
| `expert_tier` | `A` / `B` / `C` | expert tier (90 A / 100 B / 40 C in the paper's clear-reject–enriched test set) |
| `quality_score_bare` | float 0–100 | `quality_score` (**higher = better**) from the bare, label-free screener prompt (`prompts/coarse_screener_bare_v1.md`) |
| `quality_score_anchor` | float 0–100 | `quality_score` from the anchor-primed screener prompt (`prompts/coarse_screener_anchor_v1.md`) |

`reproduce.py` reports A-vs-rest AUROC on the score and C-vs-rest AUROC on the negated score.

## Building the real extract from the source workbook (for the data holders)

`tools/build_extract.py` documents the exact derivation: strip newlines/whitespace from the source
column names; keep rows whose status is one of the two labels; `grade13` and `tier` from the grade
string; `ksic2` from the 5-digit code inside the parentheses of the industry field; `desc_len_chars` from
the description; scores joined from the per-model output CSVs of the frozen set; `score_tfidf_oof` recomputed
from the description text with the protocol above; reliability and screener from the run logs. Firm ids are
re-keyed with a seeded permutation so the extract cannot be aligned with the source row order.
