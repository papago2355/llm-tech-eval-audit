---
prompt_id: coarse_screener_bare_v1
target: coarse A/B/C expert-tier screen (Section V-F, Table V-3 "bare LLM")
source: experiments/26_apprentice_evaluator/run_coarse_bare.py (SYS, prompt(), TAIL) — strings copied verbatim
decoding: temperature 0, max_tokens 1200, no seed; same 230-firm test set as coarse_screener_anchor_v1
inputs: industry code string (업종), IPC code, product name, technology name, first 1200 characters of the description — NO supervised prior
---
## System prompt (verbatim)

You are a Korean technology-evaluation expert. From the description alone, assign a coarse tier: A = top-quality (clearly fund), B = middle, C = weak (clearly reject). Respond in valid JSON only.

## User prompt template (verbatim; Python f-string fields shown in braces)

업종: {sector_code}
IPC: {ipc_code}
제품: {product_name}
기술명: {tech_name}
기술개요:
{description[:1200]}

Output JSON only: {"quality_score": <float 0-100, higher=better>, "tier": "<A|B|C>"}
