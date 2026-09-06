# Reproduction tables

Data: `data/synthetic` — firms.csv 9,926 rows, reliability.csv 1,668 ratings, screener.csv 230 rows. Bootstrap 2000 (seed 0); TOST 4000; CV seed 42.


## Table I — Dataset composition by grade tier (population)

| Grade tier | Firms | Failures | Rate |
|---|---|---|---|
| A-tier | 1,466 | 30 | 2.0% |
| B-tier | 8,370 | 432 | 5.2% |
| C-tier | 90 | 67 | 74.4% |
| Below-A (B + C) | 8,460 | 499 | 5.9% |
| Full labeled corpus | 9,926 | 529 | 5.3% |

Usable description (80–2,000 chars): 8,030 firms (403 failures) — by tier (firms/failures) A 1,256/26, B 6,711/328, C 63/49; Below-A usable: 6,774 firms (377 failures). Frozen evaluation set: 967 rows = 403 failures (incl. 26 A-tier) + 564 registry-normal; Below-A rows 941, B-tier rows 878, C-tier rows 63.


## Table V-1 — Method-invariant plateau on the Below-A frozen set

*n = 941 rows, 377 failures, 564 registry-normal. Bootstrap: 2000 row resamples, seed 0, percentile 2.5/97.5. DeLong CI is asymptotic.*

| Predictor | n | AUROC | 95% bootstrap CI | 95% DeLong CI |
|---|---|---|---|---|
| Expert grade (ordinal 0-12) | 941 | 0.614 | [0.579, 0.649] | [0.580, 0.648] |
| TF-IDF (char n-gram) OOF | 941 | 0.608 | [0.573, 0.643] | [0.572, 0.644] |
| Sonnet 4.6 | 941 | 0.597 | [0.561, 0.632] | [0.561, 0.634] |
| GPT-5.5 | 941 | 0.600 | [0.562, 0.635] | [0.563, 0.636] |
| Qwen 3.5 | 941 | 0.600 | [0.562, 0.636] | [0.564, 0.637] |
| DeepSeek V4 | 939 | 0.551 | [0.513, 0.589] | [0.513, 0.589] |
| Llama 4 Maverick | 941 | 0.530 | [0.490, 0.568] | [0.493, 0.568] |
| Length-only | 941 | 0.524 | [0.487, 0.562] | [0.486, 0.562] |


## Table V-1b — Within the B-tier (frozen set)

*n = 878 rows, 328 failures, 550 registry-normal. Bootstrap: 2000 row resamples, seed 0, percentile 2.5/97.5. DeLong CI is asymptotic.*

| Predictor | n | AUROC | 95% bootstrap CI | 95% DeLong CI |
|---|---|---|---|---|
| Expert grade (ordinal 0-12) | 878 | 0.572 | [0.536, 0.608] | [0.537, 0.608] |
| TF-IDF (char n-gram) OOF | 878 | 0.606 | [0.567, 0.643] | [0.567, 0.644] |
| Sonnet 4.6 | 878 | 0.589 | [0.551, 0.627] | [0.551, 0.628] |
| GPT-5.5 | 878 | 0.592 | [0.554, 0.629] | [0.553, 0.630] |
| Qwen 3.5 | 878 | 0.599 | [0.560, 0.636] | [0.560, 0.637] |
| DeepSeek V4 | 876 | 0.539 | [0.499, 0.579] | [0.499, 0.579] |
| Llama 4 Maverick | 878 | 0.531 | [0.492, 0.570] | [0.491, 0.570] |
| Length-only | 878 | 0.519 | [0.479, 0.558] | [0.479, 0.558] |


## Table V-1c — Pairwise DeLong tests and TOST equivalence (Below-A frozen set)

*n = 941. DeLong: two-sided test for correlated AUROCs. TOST: equivalence if the 90% paired-bootstrap CI of the AUROC difference lies within ±0.05 (4000 resamples, seed 0); p_TOST from the DeLong SE.*

| Pair | n | AUROC 1 | AUROC 2 | Δ | SE | DeLong p | 90% CI (Δ) | p_TOST | equiv. ±0.05 |
|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 vs Expert grade (ordinal 0-12) | 941 | 0.597 | 0.614 | -0.017 | 0.023 | 0.464 | [-0.055, +0.020] | 0.072 | no |
| Sonnet 4.6 vs TF-IDF (char n-gram) OOF | 941 | 0.597 | 0.608 | -0.011 | 0.024 | 0.655 | [-0.050, +0.027] | 0.049 | yes |
| Expert grade (ordinal 0-12) vs TF-IDF (char n-gram) OOF | 941 | 0.614 | 0.608 | +0.006 | 0.023 | 0.796 | [-0.032, +0.045] | 0.030 | yes |
| Sonnet 4.6 vs GPT-5.5 | 941 | 0.597 | 0.600 | -0.002 | 0.013 | 0.846 | [-0.024, +0.019] | 0.000 | yes |
| Sonnet 4.6 vs Qwen 3.5 | 941 | 0.597 | 0.600 | -0.003 | 0.014 | 0.822 | [-0.025, +0.020] | 0.000 | yes |
| Sonnet 4.6 vs DeepSeek V4 | 939 | 0.597 | 0.551 | +0.047 | 0.017 | 0.005 | [+0.019, +0.073] | 0.422 | no |
| Sonnet 4.6 vs Llama 4 Maverick | 941 | 0.597 | 0.530 | +0.067 | 0.020 | 0.001 | [+0.034, +0.100] | 0.802 | no |
| Sonnet 4.6 vs Length-only | 941 | 0.597 | 0.524 | +0.073 | 0.024 | 0.002 | [+0.034, +0.113] | 0.836 | no |
| Expert grade (ordinal 0-12) vs Length-only | 941 | 0.614 | 0.524 | +0.090 | 0.025 | 0.000 | [+0.048, +0.131] | 0.945 | no |


## Table V-1d — Pairwise DeLong tests and TOST equivalence (B-tier frozen set)

*n = 878. DeLong: two-sided test for correlated AUROCs. TOST: equivalence if the 90% paired-bootstrap CI of the AUROC difference lies within ±0.05 (4000 resamples, seed 0); p_TOST from the DeLong SE.*

| Pair | n | AUROC 1 | AUROC 2 | Δ | SE | DeLong p | 90% CI (Δ) | p_TOST | equiv. ±0.05 |
|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 vs Expert grade (ordinal 0-12) | 878 | 0.589 | 0.572 | +0.017 | 0.024 | 0.477 | [-0.023, +0.055] | 0.082 | no |
| Sonnet 4.6 vs TF-IDF (char n-gram) OOF | 878 | 0.589 | 0.606 | -0.016 | 0.025 | 0.516 | [-0.057, +0.024] | 0.088 | no |
| Expert grade (ordinal 0-12) vs TF-IDF (char n-gram) OOF | 878 | 0.572 | 0.606 | -0.033 | 0.024 | 0.162 | [-0.072, +0.005] | 0.238 | no |


## Supplement — AUROC on the full labeled population (scores available outside the frozen set)

*Present only when the extract carries LLM scores for firms outside the frozen set (synthetic data does).*

| Population | Predictor | n | Failures | AUROC | 95% CI (500 boots) |
|---|---|---|---|---|---|
| Full corpus | Expert grade (ordinal 0-12) | 9926 | 529 | 0.657 | [0.634, 0.678] |
| Full corpus | Sonnet 4.6 | 9926 | 529 | 0.635 | [0.612, 0.658] |
| Full corpus | Length-only | 9926 | 529 | 0.510 | [0.482, 0.538] |
| Below-A | Expert grade (ordinal 0-12) | 8460 | 499 | 0.635 | [0.611, 0.660] |
| Below-A | Sonnet 4.6 | 8460 | 499 | 0.616 | [0.591, 0.641] |
| Below-A | Length-only | 8460 | 499 | 0.508 | [0.481, 0.533] |
| B-tier | Expert grade (ordinal 0-12) | 8370 | 432 | 0.581 | [0.553, 0.605] |
| B-tier | Sonnet 4.6 | 8370 | 432 | 0.609 | [0.581, 0.636] |
| B-tier | Length-only | 8370 | 432 | 0.514 | [0.483, 0.544] |
| A-tier | Expert grade (ordinal 0-12) | 1466 | 30 | 0.541 | [0.450, 0.624] |
| A-tier | Sonnet 4.6 | 1466 | 30 | 0.634 | [0.536, 0.738] |
| A-tier | Length-only | 1466 | 30 | 0.560 | [0.459, 0.658] |
| C-tier | Expert grade (ordinal 0-12) | 90 | 67 | 0.548 | [0.421, 0.664] |
| C-tier | Sonnet 4.6 | 90 | 67 | 0.610 | [0.471, 0.746] |
| C-tier | Length-only | 90 | 67 | 0.726 | [0.610, 0.838] |


## Table V-2 — Cross-model style profiles (frozen set)

*n = 967; fail-rate = share with prob_fail ≥ 50; Spearman ρ vs Sonnet on pairwise-complete rows.*

| Model | n | Mean | SD | Fail-rate | ρ vs Sonnet |
|---|---|---|---|---|---|
| Sonnet 4.6 | 967 | 55.5 | 12.6 | 70% | 1.00 |
| GPT-5.5 | 967 | 50.0 | 13.4 | 49% | 0.79 |
| Qwen 3.5 | 967 | 52.6 | 21.7 | 55% | 0.75 |
| DeepSeek V4 | 965 | 59.6 | 17.3 | 73% | 0.62 |
| Llama 4 Maverick | 967 | 48.1 | 23.1 | 48% | 0.43 |


## Table V-4 — Test–retest reliability of the judge on byte-identical descriptions

*One-way random-effects ICC: ICC(1) single rating, ICC(1,k) mean of k re-scorings; F-distribution CI (exact) and firm-level cluster-bootstrap CI; SEM = √MSW; SDC95 = 1.96·√2·SEM; spans are max−min over re-scorings.*

| Temp. | Firms | Ratings | k₀ | ICC(1) | F-CI | boot-CI | ICC(1,k) | F-CI | boot-CI | SEM | SDC₉₅ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 139 | 1390 | 10.0 | 0.986 | [0.982, 0.989] | [0.981, 0.990] | 0.999 | [0.998, 0.999] | [0.998, 0.999] | 1.79 | 5.0 |
| 0 | 139 | 278 | 2.0 | 0.995 | [0.993, 0.996] | [0.989, 0.999] | 0.997 | [0.996, 0.998] | [0.994, 1.000] | 1.10 | 3.0 |

Rep-to-rep agreement (share of firms):

| Temp. | Firms | Identical | ≤5 pts | ≤10 pts | ≤20 pts | Span median | Span mean | Span max | Verdict change (≥50) |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 139 | 58% | 63% | 94% | 100% | 0.0 | 3.5 | 13 | 5.0% |
| 0 | 139 | 95% | 98% | 99% | 100% | 0.0 | 0.3 | 12 | 0.0% |

ICC(1) by expert tier:

| Temp. | Tier | Firms | ICC(1) | SEM |
|---|---|---|---|---|
| 0.7 | A | 50 | 0.984 | 1.85 |
| 0.7 | B | 50 | 0.985 | 1.71 |
| 0.7 | C | 39 | 0.984 | 1.82 |
| 0 | A | 50 | 0.996 | 0.92 |
| 0 | B | 50 | 0.997 | 0.80 |
| 0 | C | 39 | 0.989 | 1.55 |


## Table OP — Operating points on the Below-A frozen set

*Flag = prob_fail ≥ 50. Capture = sensitivity; burden = share of registry-normal firms flagged. PPV/NPV re-weighted to the population Below-A prevalence 5.90% (Bayes, from sensitivity/specificity).*

| Model | n | TP | FP | FN | TN | Capture | Burden | PPV (sample) | PPV (pop.) | NPV (pop.) | Flag rate (pop.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 | 941 | 296 | 368 | 81 | 196 | 78.5% | 65.2% | 44.6% | 7.0% | 96.27% | 66.0% |
| GPT-5.5 | 941 | 215 | 250 | 162 | 314 | 57.0% | 44.3% | 46.2% | 7.5% | 95.39% | 45.1% |
| Qwen 3.5 | 941 | 236 | 284 | 141 | 280 | 62.6% | 50.4% | 45.4% | 7.2% | 95.49% | 51.1% |
| DeepSeek V4 | 939 | 280 | 407 | 97 | 155 | 74.3% | 72.4% | 40.8% | 6.0% | 94.48% | 72.5% |
| Llama 4 Maverick | 941 | 200 | 257 | 177 | 307 | 53.1% | 45.6% | 43.8% | 6.8% | 94.87% | 46.0% |


## Table SEC — Sector-stratified AUROC (Below-A frozen set)

*Pooled within-ksic2 AUROC = P(score_fail > score_normal | same 2-digit division), pair-weighted over divisions with both classes present; compared with the unstratified AUROC on the rows used.*

| Predictor | Rows used | Divisions | Unstratified AUROC | Within-division pooled AUROC | Δ |
|---|---|---|---|---|---|
| Expert grade (ordinal 0-12) | 893 | 63 | 0.614 | 0.622 | +0.008 |
| TF-IDF (char n-gram) OOF | 893 | 63 | 0.608 | 0.604 | -0.004 |
| Sonnet 4.6 | 893 | 63 | 0.597 | 0.593 | -0.005 |
| GPT-5.5 | 893 | 63 | 0.600 | 0.605 | +0.006 |
| Qwen 3.5 | 893 | 63 | 0.600 | 0.591 | -0.009 |

Reference: leave-one-out sector failure-rate prior (no text) — AUROC 0.588 on 934 rows.


## Table TRI — Cross-validated two-sided triage (Sonnet score, Below-A frozen set)

*5-fold stratified CV (seed 42): on each training fold, clear-accept = largest cut-off whose band failure rate is ≤ 0.5× the training prevalence; clear-reject = smallest cut-off whose band failure rate is ≥ 1.5× the prevalence (bands ≥ 20 firms). Routing is evaluated out-of-fold. False accepts = failures routed to accept; false rejects = registry-normal firms routed to reject.*

| Predictor | n | Accept (OOF) | Review (OOF) | Reject (OOF) | False accepts | False rejects | Fail-rate accept | Fail-rate reject | Routing at pop. prevalence (acc/rev/rej) | Accept-band fail-rate (pop.) | Reject-band fail-rate (pop.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 | 941 | 0.0% (0) | 98.0% (922) | 2.0% (19) | 0 / 377 | 11 / 564 | — | 42.1% | 0.0% / 98.0% / 2.0% | — | 6.4% |
| TF-IDF (char n-gram) OOF | 941 | 8.6% (81) | 81.5% (767) | 9.9% (93) | 18 / 377 | 39 / 564 | 22.2% | 58.1% | 10.8% / 81.9% / 7.4% | 2.61% | 11.5% |
| Expert grade (ordinal 0-12) | 941 | 0.0% (0) | 93.3% (878) | 6.7% (63) | 0 / 377 | 14 / 564 | — | 77.8% | 0.0% / 96.9% / 3.1% | — | 24.7% |


## Table V-3 — Two-sided coarse screen (A-vs-rest, C-vs-rest)

*n = 230 (90 A / 100 B / 40 C). Quality score is higher-is-better; C-vs-rest uses the negated score. Bootstrap CIs as above.*

| Method | n | A-vs-rest AUROC | 95% CI | C-vs-rest AUROC | 95% CI |
|---|---|---|---|---|---|
| Bare (label-free) LLM | 230 | 0.680 | [0.612, 0.748] | 0.636 | [0.545, 0.735] |
| Anchor-primed LLM | 230 | 0.713 | [0.639, 0.779] | 0.708 | [0.615, 0.791] |
