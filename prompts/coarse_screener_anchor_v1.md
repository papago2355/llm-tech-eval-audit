---
prompt_id: coarse_screener_anchor_v1
target: coarse A/B/C expert-tier screen (Section V-F, TABLE 7 "anchor-primed LLM")
source: experiments/26_apprentice_evaluator/run_coarse.py (SYS, prompt(), TAIL) — strings copied verbatim
decoding: temperature 0, max_tokens 1400, no seed; models anthropic/claude-sonnet-4.6 and openai/gpt-oss-20b (reasoning effort "low" for gpt-oss)
inputs: industry code string (업종), IPC code, product name, technology name, first 1200 characters of the description;
        plus a SUPERVISED prior distilled from the training pool — the sector's typical tier and 15 signal words each for A-typical and C-typical text
test set: 230 firms (90 A / 100 B / 40 C), stratified sample with seed 11 from the pool of graded firms with 60-2000-char descriptions
---
## System prompt (verbatim)

You are a Korean technology-evaluation expert. Assign a coarse tier: A = top-quality (clearly fund), B = middle, C = weak (clearly reject). Respond in valid JSON only.

## User prompt template (verbatim; Python f-string fields shown in braces)

Data on thousands of past evaluations: this industry (업종={sector_code}) is typically tier ~{sector_typical_tier}. Terms common in TOP (A) techs: {up_terms}. Terms common in WEAK (C) techs: {down_terms}.

Grade this technology (anchor to the industry norm; use the text to move up/down):
업종: {sector_code}
IPC: {ipc_code}
제품(product): {product_name}
기술명(tech name): {tech_name}
기술개요:
{description[:1200]}

Output JSON only: {"quality_score": <float 0-100, higher=better>, "tier": "<A|B|C>", "rationale": "<one sentence>"}

## How the anchor is built (run_coarse.knowledge)

* `sector_typical_tier`: the pool's mean 13-level grade index within the firm's industry code, rounded and mapped to A/B/C (pool mean if the code is unseen).
* `up_terms` / `down_terms`: CountVectorizer(token_pattern=[가-힣A-Za-z]{2,}, min_df=6, max_features=3000) on the pool; the 15 tokens with the largest / smallest log((df_A + .005)/(df_rest + .005)).
* The test set is disjoint from the pool.
