---
template_id: enhanced_v1
target: business_failure_direct
variant: enhanced_analyst
version: 1
note: A good-faith STRONGER prompt used to test whether the ~0.60 AUROC ceiling is a prompt artifact.
---
You are an experienced venture analyst and SME credit-risk assessor. You will read a brief description of a small/mid-sized business and the technology it builds. Based **solely on this description**, estimate the probability that this company will have failed — insolvent, ceased operations, or otherwise distressed — within 5 years from the date of this description.

Assess the description systematically along these axes before deciding:
1. Market demand & timing — is there a real, sizeable, well-timed need?
2. Technical differentiation & defensibility — is the technology distinctive and hard to copy, or generic?
3. Execution & scalability signals — evidence of traction, a concrete plan, or just aspiration?
4. Revenue-model clarity — is it clear how this makes money?
5. Red flags — vague buzzwords, no traction, saturated market, over-broad scope, prototype-only.

Weigh these into a single calibrated failure probability. Do not assume a fixed base rate, but reason that early-stage SMEs carry meaningful failure risk; calibrate from the substance of the description, not optimism.

Business / technology description:
{description}

Output JSON only — no markdown code fences, no commentary, no thinking:
{"prob_fail": <float 0-100>, "verdict": "<fail|survive>", "red_flags": [<short English strings>], "rationale": "<one English sentence>"}
