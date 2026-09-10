# Supplementary Table S-1 in full: all pairwise AUROC comparisons on the real frozen set

This document accompanies Supplementary Table S-1 of the manuscript, which prints the twelve comparisons that carry the paper's claims; here every pair of the ten predictors is listed (45 pairs per subset). Computed on the real, proprietary frozen evaluation set, so it is shipped as a static result; the de-identified extract does not carry the multilingual-e5 or fresh-Sonnet scores, so these pairs cannot be regenerated from `analysis/reproduce.py` (which recomputes the pairs among the eight predictors it holds).

Settings: paired bootstrap 2000 resamples (seed 0; same draws for both predictors of a pair); DeLong test after Sun & Xu (2014) midrank placement values; TOST equivalence if the 90% paired-bootstrap CI of Δ lies within ±0.05. Δ = AUROC(A) − AUROC(B). 'Detectable difference' = DeLong p < 0.05.

## Below-A (n = 941; 377 failures, 564 registry-normal) — 45 pairs

| Predictor A | Predictor B | AUROC A | AUROC B | Δ | DeLong p | 90% bootstrap CI of Δ | 95% bootstrap CI of Δ | TOST-equivalent (±0.05) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 (cached panel run) | Sonnet 4.6 (fresh control run) | 0.604 | 0.603 | +0.001 | 0.796 | [-0.007, +0.010] | [-0.009, +0.012] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | GPT-5.5 | 0.604 | 0.599 | +0.005 | 0.659 | [-0.015, +0.026] | [-0.019, +0.030] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | Qwen 3.5 | 0.604 | 0.597 | +0.007 | 0.563 | [-0.013, +0.028] | [-0.017, +0.031] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | DeepSeek V4 | 0.603 | 0.550 | +0.053 | 0.001 | [+0.027, +0.079] | [+0.022, +0.084] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (cached panel run) | Llama 4 Maverick | 0.604 | 0.526 | +0.077 | 0.000 | [+0.047, +0.111] | [+0.041, +0.116] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (cached panel run) | TF-IDF (char_wb 2–4, OOF) | 0.604 | 0.609 | -0.005 | 0.832 | [-0.043, +0.034] | [-0.050, +0.040] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | multilingual-e5-base (OOF) | 0.604 | 0.598 | +0.006 | 0.783 | [-0.028, +0.039] | [-0.035, +0.046] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | Expert grade (ordinal) | 0.604 | 0.616 | -0.012 | 0.594 | [-0.047, +0.024] | [-0.054, +0.032] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | Length-only | 0.604 | 0.524 | +0.080 | 0.001 | [+0.040, +0.118] | [+0.031, +0.124] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | GPT-5.5 | 0.603 | 0.599 | +0.004 | 0.745 | [-0.016, +0.024] | [-0.020, +0.028] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | Qwen 3.5 | 0.603 | 0.597 | +0.006 | 0.645 | [-0.015, +0.026] | [-0.018, +0.030] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | DeepSeek V4 | 0.602 | 0.550 | +0.052 | 0.001 | [+0.025, +0.077] | [+0.021, +0.081] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | Llama 4 Maverick | 0.603 | 0.526 | +0.076 | 0.000 | [+0.045, +0.109] | [+0.039, +0.114] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | TF-IDF (char_wb 2–4, OOF) | 0.603 | 0.609 | -0.006 | 0.786 | [-0.044, +0.032] | [-0.051, +0.038] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | multilingual-e5-base (OOF) | 0.603 | 0.598 | +0.004 | 0.837 | [-0.029, +0.038] | [-0.036, +0.044] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | Expert grade (ordinal) | 0.603 | 0.616 | -0.013 | 0.547 | [-0.048, +0.023] | [-0.054, +0.031] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | Length-only | 0.603 | 0.524 | +0.079 | 0.001 | [+0.038, +0.116] | [+0.032, +0.123] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | Qwen 3.5 | 0.599 | 0.597 | +0.002 | 0.873 | [-0.019, +0.023] | [-0.024, +0.026] | yes | TOST-equivalent at ±0.05 |
| GPT-5.5 | DeepSeek V4 | 0.599 | 0.550 | +0.049 | 0.004 | [+0.020, +0.076] | [+0.014, +0.082] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | Llama 4 Maverick | 0.599 | 0.526 | +0.072 | 0.000 | [+0.041, +0.105] | [+0.035, +0.110] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | TF-IDF (char_wb 2–4, OOF) | 0.599 | 0.609 | -0.010 | 0.671 | [-0.051, +0.029] | [-0.060, +0.036] | no | no clear difference detected, equivalence not established |
| GPT-5.5 | multilingual-e5-base (OOF) | 0.599 | 0.598 | +0.000 | 0.985 | [-0.035, +0.034] | [-0.042, +0.041] | yes | TOST-equivalent at ±0.05 |
| GPT-5.5 | Expert grade (ordinal) | 0.599 | 0.616 | -0.017 | 0.442 | [-0.054, +0.021] | [-0.062, +0.028] | no | no clear difference detected, equivalence not established |
| GPT-5.5 | Length-only | 0.599 | 0.524 | +0.075 | 0.002 | [+0.033, +0.113] | [+0.028, +0.120] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | DeepSeek V4 | 0.598 | 0.550 | +0.047 | 0.005 | [+0.019, +0.075] | [+0.014, +0.081] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | Llama 4 Maverick | 0.597 | 0.526 | +0.070 | 0.001 | [+0.039, +0.103] | [+0.033, +0.108] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | TF-IDF (char_wb 2–4, OOF) | 0.597 | 0.609 | -0.012 | 0.601 | [-0.051, +0.026] | [-0.059, +0.032] | no | no clear difference detected, equivalence not established |
| Qwen 3.5 | multilingual-e5-base (OOF) | 0.597 | 0.598 | -0.002 | 0.941 | [-0.036, +0.032] | [-0.044, +0.039] | yes | TOST-equivalent at ±0.05 |
| Qwen 3.5 | Expert grade (ordinal) | 0.597 | 0.616 | -0.019 | 0.402 | [-0.058, +0.019] | [-0.065, +0.025] | no | no clear difference detected, equivalence not established |
| Qwen 3.5 | Length-only | 0.597 | 0.524 | +0.073 | 0.002 | [+0.035, +0.109] | [+0.030, +0.117] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | Llama 4 Maverick | 0.550 | 0.527 | +0.023 | 0.233 | [-0.008, +0.054] | [-0.013, +0.060] | no | no clear difference detected, equivalence not established |
| DeepSeek V4 | TF-IDF (char_wb 2–4, OOF) | 0.550 | 0.608 | -0.058 | 0.022 | [-0.098, -0.016] | [-0.106, -0.008] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | multilingual-e5-base (OOF) | 0.550 | 0.598 | -0.048 | 0.041 | [-0.086, -0.009] | [-0.092, -0.002] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | Expert grade (ordinal) | 0.550 | 0.615 | -0.065 | 0.005 | [-0.102, -0.027] | [-0.110, -0.019] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | Length-only | 0.550 | 0.523 | +0.028 | 0.296 | [-0.014, +0.071] | [-0.022, +0.080] | no | no clear difference detected, equivalence not established |
| Llama 4 Maverick | TF-IDF (char_wb 2–4, OOF) | 0.526 | 0.609 | -0.082 | 0.002 | [-0.127, -0.039] | [-0.134, -0.031] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | multilingual-e5-base (OOF) | 0.526 | 0.598 | -0.072 | 0.006 | [-0.115, -0.031] | [-0.122, -0.024] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | Expert grade (ordinal) | 0.526 | 0.616 | -0.089 | 0.000 | [-0.128, -0.050] | [-0.135, -0.042] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | Length-only | 0.526 | 0.524 | +0.003 | 0.930 | [-0.046, +0.050] | [-0.058, +0.057] | no | no clear difference detected, equivalence not established |
| TF-IDF (char_wb 2–4, OOF) | multilingual-e5-base (OOF) | 0.609 | 0.598 | +0.011 | 0.576 | [-0.021, +0.042] | [-0.027, +0.047] | yes | TOST-equivalent at ±0.05 |
| TF-IDF (char_wb 2–4, OOF) | Expert grade (ordinal) | 0.609 | 0.616 | -0.007 | 0.778 | [-0.047, +0.036] | [-0.054, +0.042] | yes | TOST-equivalent at ±0.05 |
| TF-IDF (char_wb 2–4, OOF) | Length-only | 0.609 | 0.524 | +0.085 | 0.000 | [+0.046, +0.124] | [+0.037, +0.130] | no | detectable difference (DeLong p<0.05) |
| multilingual-e5-base (OOF) | Expert grade (ordinal) | 0.598 | 0.616 | -0.018 | 0.463 | [-0.055, +0.022] | [-0.064, +0.028] | no | no clear difference detected, equivalence not established |
| multilingual-e5-base (OOF) | Length-only | 0.598 | 0.524 | +0.074 | 0.001 | [+0.037, +0.110] | [+0.030, +0.116] | no | detectable difference (DeLong p<0.05) |
| Expert grade (ordinal) | Length-only | 0.616 | 0.524 | +0.092 | 0.000 | [+0.049, +0.134] | [+0.039, +0.141] | no | detectable difference (DeLong p<0.05) |

Tally: 16 of 45 pairs TOST-equivalent at ±0.05; 21 of 45 pairs with a detectable DeLong difference (p < 0.05); 8 neither (inconclusive).

Single-predictor AUROCs used above:

| Predictor | AUROC | 95% bootstrap CI |
|---|---|---|
| Sonnet 4.6 (cached panel run) | 0.604 | [0.569, 0.640] |
| Sonnet 4.6 (fresh control run) | 0.603 | [0.568, 0.639] |
| GPT-5.5 | 0.599 | [0.562, 0.636] |
| Qwen 3.5 | 0.597 | [0.561, 0.633] |
| DeepSeek V4 | 0.550 | [0.516, 0.587] |
| Llama 4 Maverick | 0.526 | [0.491, 0.562] |
| TF-IDF (char_wb 2–4, OOF) | 0.609 | [0.572, 0.646] |
| multilingual-e5-base (OOF) | 0.598 | [0.564, 0.634] |
| Expert grade (ordinal) | 0.616 | [0.581, 0.652] |
| Length-only | 0.524 | [0.488, 0.562] |

## B-tier (n = 878; 328 failures, 550 registry-normal) — 45 pairs

| Predictor A | Predictor B | AUROC A | AUROC B | Δ | DeLong p | 90% bootstrap CI of Δ | 95% bootstrap CI of Δ | TOST-equivalent (±0.05) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Sonnet 4.6 (cached panel run) | Sonnet 4.6 (fresh control run) | 0.588 | 0.586 | +0.002 | 0.761 | [-0.008, +0.011] | [-0.009, +0.013] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | GPT-5.5 | 0.588 | 0.587 | +0.001 | 0.942 | [-0.019, +0.022] | [-0.023, +0.025] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | Qwen 3.5 | 0.588 | 0.585 | +0.003 | 0.816 | [-0.019, +0.025] | [-0.025, +0.030] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | DeepSeek V4 | 0.588 | 0.539 | +0.049 | 0.004 | [+0.021, +0.077] | [+0.016, +0.082] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (cached panel run) | Llama 4 Maverick | 0.588 | 0.515 | +0.073 | 0.001 | [+0.037, +0.106] | [+0.030, +0.114] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (cached panel run) | TF-IDF (char_wb 2–4, OOF) | 0.588 | 0.602 | -0.014 | 0.578 | [-0.056, +0.026] | [-0.063, +0.036] | no | no clear difference detected, equivalence not established |
| Sonnet 4.6 (cached panel run) | multilingual-e5-base (OOF) | 0.588 | 0.595 | -0.008 | 0.731 | [-0.045, +0.029] | [-0.052, +0.036] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (cached panel run) | Expert grade (ordinal) | 0.588 | 0.573 | +0.015 | 0.529 | [-0.023, +0.053] | [-0.030, +0.063] | no | no clear difference detected, equivalence not established |
| Sonnet 4.6 (cached panel run) | Length-only | 0.588 | 0.526 | +0.062 | 0.014 | [+0.021, +0.102] | [+0.013, +0.111] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | GPT-5.5 | 0.586 | 0.587 | -0.001 | 0.951 | [-0.022, +0.020] | [-0.026, +0.024] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | Qwen 3.5 | 0.586 | 0.585 | +0.001 | 0.919 | [-0.022, +0.024] | [-0.025, +0.029] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | DeepSeek V4 | 0.586 | 0.539 | +0.047 | 0.005 | [+0.020, +0.075] | [+0.015, +0.081] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | Llama 4 Maverick | 0.586 | 0.515 | +0.071 | 0.001 | [+0.034, +0.105] | [+0.027, +0.113] | no | detectable difference (DeLong p<0.05) |
| Sonnet 4.6 (fresh control run) | TF-IDF (char_wb 2–4, OOF) | 0.586 | 0.602 | -0.016 | 0.532 | [-0.057, +0.026] | [-0.065, +0.033] | no | no clear difference detected, equivalence not established |
| Sonnet 4.6 (fresh control run) | multilingual-e5-base (OOF) | 0.586 | 0.595 | -0.009 | 0.679 | [-0.046, +0.028] | [-0.053, +0.036] | yes | TOST-equivalent at ±0.05 |
| Sonnet 4.6 (fresh control run) | Expert grade (ordinal) | 0.586 | 0.573 | +0.013 | 0.576 | [-0.026, +0.052] | [-0.033, +0.059] | no | no clear difference detected, equivalence not established |
| Sonnet 4.6 (fresh control run) | Length-only | 0.586 | 0.526 | +0.060 | 0.018 | [+0.019, +0.101] | [+0.010, +0.109] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | Qwen 3.5 | 0.587 | 0.585 | +0.002 | 0.868 | [-0.020, +0.023] | [-0.026, +0.028] | yes | TOST-equivalent at ±0.05 |
| GPT-5.5 | DeepSeek V4 | 0.586 | 0.539 | +0.048 | 0.008 | [+0.018, +0.077] | [+0.011, +0.082] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | Llama 4 Maverick | 0.587 | 0.515 | +0.072 | 0.001 | [+0.036, +0.105] | [+0.030, +0.112] | no | detectable difference (DeLong p<0.05) |
| GPT-5.5 | TF-IDF (char_wb 2–4, OOF) | 0.587 | 0.602 | -0.015 | 0.571 | [-0.058, +0.028] | [-0.068, +0.038] | no | no clear difference detected, equivalence not established |
| GPT-5.5 | multilingual-e5-base (OOF) | 0.587 | 0.595 | -0.009 | 0.712 | [-0.047, +0.028] | [-0.054, +0.035] | yes | TOST-equivalent at ±0.05 |
| GPT-5.5 | Expert grade (ordinal) | 0.587 | 0.573 | +0.014 | 0.559 | [-0.025, +0.052] | [-0.031, +0.058] | no | no clear difference detected, equivalence not established |
| GPT-5.5 | Length-only | 0.587 | 0.526 | +0.061 | 0.017 | [+0.020, +0.101] | [+0.012, +0.107] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | DeepSeek V4 | 0.585 | 0.539 | +0.046 | 0.010 | [+0.016, +0.075] | [+0.012, +0.081] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | Llama 4 Maverick | 0.585 | 0.515 | +0.070 | 0.001 | [+0.033, +0.105] | [+0.025, +0.113] | no | detectable difference (DeLong p<0.05) |
| Qwen 3.5 | TF-IDF (char_wb 2–4, OOF) | 0.585 | 0.602 | -0.017 | 0.497 | [-0.058, +0.026] | [-0.066, +0.035] | no | no clear difference detected, equivalence not established |
| Qwen 3.5 | multilingual-e5-base (OOF) | 0.585 | 0.595 | -0.011 | 0.629 | [-0.048, +0.025] | [-0.055, +0.032] | yes | TOST-equivalent at ±0.05 |
| Qwen 3.5 | Expert grade (ordinal) | 0.585 | 0.573 | +0.012 | 0.635 | [-0.028, +0.051] | [-0.037, +0.058] | no | no clear difference detected, equivalence not established |
| Qwen 3.5 | Length-only | 0.585 | 0.526 | +0.059 | 0.015 | [+0.020, +0.097] | [+0.013, +0.104] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | Llama 4 Maverick | 0.539 | 0.515 | +0.024 | 0.246 | [-0.010, +0.058] | [-0.017, +0.064] | no | no clear difference detected, equivalence not established |
| DeepSeek V4 | TF-IDF (char_wb 2–4, OOF) | 0.539 | 0.601 | -0.063 | 0.018 | [-0.105, -0.019] | [-0.111, -0.009] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | multilingual-e5-base (OOF) | 0.539 | 0.595 | -0.056 | 0.022 | [-0.094, -0.016] | [-0.100, -0.009] | no | detectable difference (DeLong p<0.05) |
| DeepSeek V4 | Expert grade (ordinal) | 0.539 | 0.574 | -0.035 | 0.151 | [-0.076, +0.004] | [-0.083, +0.013] | no | no clear difference detected, equivalence not established |
| DeepSeek V4 | Length-only | 0.539 | 0.525 | +0.013 | 0.625 | [-0.031, +0.059] | [-0.040, +0.066] | no | no clear difference detected, equivalence not established |
| Llama 4 Maverick | TF-IDF (char_wb 2–4, OOF) | 0.515 | 0.602 | -0.087 | 0.002 | [-0.132, -0.040] | [-0.141, -0.031] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | multilingual-e5-base (OOF) | 0.515 | 0.595 | -0.080 | 0.004 | [-0.126, -0.036] | [-0.135, -0.026] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | Expert grade (ordinal) | 0.515 | 0.573 | -0.058 | 0.021 | [-0.099, -0.017] | [-0.108, -0.009] | no | detectable difference (DeLong p<0.05) |
| Llama 4 Maverick | Length-only | 0.515 | 0.526 | -0.011 | 0.726 | [-0.061, +0.040] | [-0.071, +0.048] | no | no clear difference detected, equivalence not established |
| TF-IDF (char_wb 2–4, OOF) | multilingual-e5-base (OOF) | 0.602 | 0.595 | +0.006 | 0.756 | [-0.029, +0.038] | [-0.035, +0.045] | yes | TOST-equivalent at ±0.05 |
| TF-IDF (char_wb 2–4, OOF) | Expert grade (ordinal) | 0.602 | 0.573 | +0.029 | 0.259 | [-0.013, +0.070] | [-0.020, +0.080] | no | no clear difference detected, equivalence not established |
| TF-IDF (char_wb 2–4, OOF) | Length-only | 0.602 | 0.526 | +0.076 | 0.003 | [+0.034, +0.119] | [+0.026, +0.125] | no | detectable difference (DeLong p<0.05) |
| multilingual-e5-base (OOF) | Expert grade (ordinal) | 0.595 | 0.573 | +0.022 | 0.372 | [-0.018, +0.063] | [-0.026, +0.072] | no | no clear difference detected, equivalence not established |
| multilingual-e5-base (OOF) | Length-only | 0.595 | 0.526 | +0.069 | 0.002 | [+0.033, +0.107] | [+0.025, +0.114] | no | detectable difference (DeLong p<0.05) |
| Expert grade (ordinal) | Length-only | 0.573 | 0.526 | +0.047 | 0.082 | [+0.001, +0.092] | [-0.006, +0.101] | no | no clear difference detected, equivalence not established |

Tally: 11 of 45 pairs TOST-equivalent at ±0.05; 19 of 45 pairs with a detectable DeLong difference (p < 0.05); 15 neither (inconclusive).

Single-predictor AUROCs used above:

| Predictor | AUROC | 95% bootstrap CI |
|---|---|---|
| Sonnet 4.6 (cached panel run) | 0.588 | [0.551, 0.626] |
| Sonnet 4.6 (fresh control run) | 0.586 | [0.549, 0.624] |
| GPT-5.5 | 0.587 | [0.549, 0.626] |
| Qwen 3.5 | 0.585 | [0.547, 0.622] |
| DeepSeek V4 | 0.539 | [0.502, 0.578] |
| Llama 4 Maverick | 0.515 | [0.478, 0.554] |
| TF-IDF (char_wb 2–4, OOF) | 0.602 | [0.562, 0.642] |
| multilingual-e5-base (OOF) | 0.595 | [0.560, 0.635] |
| Expert grade (ordinal) | 0.573 | [0.537, 0.610] |
| Length-only | 0.526 | [0.490, 0.566] |

