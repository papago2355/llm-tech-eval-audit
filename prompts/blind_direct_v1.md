---
template_id: blind_direct_v1
target: business_failure_direct
variant: deployed (terse) prompt — Appendix A.1 of the manuscript
version: 1
decoding: temperature 0, seed 42, max_tokens 400, reasoning disabled (temperature 0.7 without seed for the reliability stress test)
input: {description} = the raw technology description text only (no firm name, sector, grade, year or outcome)
note: the body below is byte-identical to BLIND_DIRECT in experiments/25_reliability_retest/run.py; scoring/score_openrouter.py embeds the same string.
---
You will read a brief description of a small/mid-sized business and the technology it builds. Based **solely on this description**, estimate the probability that this company will have failed — meaning it has become insolvent, ceased operations, or otherwise entered a distressed state — within 5 years from the date of this description.

Then produce a verdict ("fail" or "survive") and, if you see any, list up to 3 specific red flags that you identified in the description itself.

Do not assume any base rate, prior, or external knowledge about what fraction of submissions typically fail. Reason only from the content of the description.

Business / technology description:
{description}

Output JSON only — no markdown code fences, no commentary, no thinking:
{"prob_fail": <float 0-100>, "verdict": "<fail|survive>", "red_flags": [<short English strings>], "rationale": "<one English sentence>"}
