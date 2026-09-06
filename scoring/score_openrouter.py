"""score_openrouter.py — the exact LLM scoring client used in the paper (OpenRouter chat completions).

Scores technology descriptions with the deployed failure-probability prompt (blind_direct_v1) or the
stronger analyst prompt (enhanced_v1), or any template file containing ``{description}``.

Protocol (identical to the paper's runs)
----------------------------------------
  request body : {"model": <model>, "messages": [system, user], "temperature": 0, "max_tokens": 400,
                  "seed": 42, "reasoning": {"enabled": false}}
  system prompt: prompts/system_prompt.md (verbatim)
  user prompt  : prompts/<template>.md body with {description} replaced by the raw description text
  parsing      : strip ``` fences, take the first {...} block, json.loads, float(prob_fail);
                 unparsable outputs are recorded with an ``error`` field and excluded from analysis
  retries      : up to 4 attempts with 2s, 4s, 6s, 8s back-off
  concurrency  : ThreadPoolExecutor, at most 6 workers
  resume       : one JSON record per call appended to --out; (id, temperature, rep) already present are skipped

Reliability protocol of Section V-C: ``--reps 10 --temperature 0.7`` (no seed) and ``--reps 2 --temperature 0``
(seed 42 + rep) on the same descriptions.

Input CSV: columns ``id`` and ``description`` (UTF-8). Nothing else is sent to the API.
API key   : OPENROUTER_API_KEY in the environment or in a ``.env`` file (KEY=VALUE lines) given by --env.

Usage
-----
  python scoring/score_openrouter.py --input my_descriptions.csv --out scores.jsonl
  python scoring/score_openrouter.py --input my.csv --out rel.jsonl --temperature 0.7 --reps 10 --no-seed
  python scoring/score_openrouter.py --to-csv scores.jsonl scores.csv      # flatten a jsonl to id,prob_fail,verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
MAX_TOKENS = 400
SEED = 42

# Verbatim system prompt (experiments/25_reliability_retest/run.py SYSTEM; experiments/24 SYSTEM_PROMPT)
SYSTEM = (
    "You are a careful analyst evaluating business descriptions. "
    "Respond in valid JSON only — no markdown, no commentary, no thinking."
)

# Verbatim deployed user prompt (experiments/25_reliability_retest/run.py BLIND_DIRECT), template id blind_direct_v1
BLIND_DIRECT = (
    "You will read a brief description of a small/mid-sized business and the technology it "
    "builds. Based **solely on this description**, estimate the probability that this company "
    "will have failed — meaning it has become insolvent, ceased operations, or otherwise "
    "entered a distressed state — within 5 years from the date of this description.\n\n"
    'Then produce a verdict ("fail" or "survive") and, if you see any, list up to 3 specific '
    "red flags that you identified in the description itself.\n\n"
    "Do not assume any base rate, prior, or external knowledge about what fraction of "
    "submissions typically fail. Reason only from the content of the description.\n\n"
    "Business / technology description:\n{description}\n\n"
    'Output JSON only — no markdown code fences, no commentary, no thinking:\n'
    '{"prob_fail": <float 0-100>, "verdict": "<fail|survive>", '
    '"red_flags": [<short English strings>], "rationale": "<one English sentence>"}'
)


def load_env(path: Path | None) -> str:
    """Read KEY=VALUE lines from a .env file into the environment (never printed) and return the key."""
    if path and path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found (set it in the environment or pass --env path/to/.env)")
    return key


def load_template(name_or_path: str) -> tuple[str, str]:
    """Return (template_id, template_text). Markdown files may carry a '---' front-matter block."""
    if name_or_path == "blind_direct_v1":
        return name_or_path, BLIND_DIRECT
    p = Path(name_or_path)
    if not p.exists():
        p = PROMPTS / f"{name_or_path}.md"
    raw = p.read_text()
    if raw.startswith("---"):
        _, _, body = raw.split("---", 2)
        raw = body
    return p.stem, raw.strip()


def parse_json(text: str | None):
    """Strip code fences, take the first {...} block, json.loads. Returns None on failure."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call(key: str, model: str, prompt: str, temperature: float, seed: int | None,
         max_tokens: int = MAX_TOKENS, reasoning_enabled: bool = False, timeout: float = 120.0) -> dict:
    import httpx

    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": reasoning_enabled},
    }
    if seed is not None:
        body["seed"] = seed
    last = None
    for attempt in range(4):
        try:
            t0 = time.time()
            r = httpx.post(OPENROUTER_URL, json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(f"api: {j['error']}")
            txt = j["choices"][0]["message"].get("content", "")
            u = j.get("usage", {}) or {}
            return {"text": txt, "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0),
                    "latency_ms": (time.time() - t0) * 1000, "provider_model": j.get("model")}
        except Exception as e:  # noqa: BLE001
            last = str(e)[:300]
            time.sleep(2 * (attempt + 1))
    return {"error": last}


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((str(r["id"]), float(r["temperature"]), int(r["rep"])))
            except Exception:  # noqa: BLE001
                continue
    return keys


def read_input(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "id" not in rows[0] or "description" not in rows[0]:
        sys.exit("input CSV must have columns: id, description")
    return [r for r in rows if str(r["description"]).strip()]


def run(args):
    key = load_env(args.env)
    template_id, template = load_template(args.prompt)
    rows = read_input(args.input)
    done = done_keys(args.out)
    tasks = []
    for r in rows:
        prompt = template.replace("{description}", str(r["description"]).strip())
        for rep in range(args.reps):
            k = (str(r["id"]), float(args.temperature), rep)
            if k in done:
                continue
            seed = None if args.no_seed else args.seed + rep
            tasks.append(dict(id=str(r["id"]), prompt=prompt, rep=rep, seed=seed))
    print(f"model={args.model} prompt={template_id} temperature={args.temperature} reps={args.reps} | "
          f"rows={len(rows)} done={len(done)} todo={len(tasks)}", flush=True)
    if not tasks:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=min(args.workers, 6)) as ex:
        futs = {ex.submit(call, key, args.model, t["prompt"], args.temperature, t["seed"], args.max_tokens,
                          args.reasoning): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            parsed = parse_json(res.get("text"))
            rec = {"id": t["id"], "temperature": args.temperature, "rep": t["rep"], "seed": t["seed"],
                   "model": args.model, "prompt_id": template_id, "prob_fail": None, "verdict": None,
                   "red_flags": None, "rationale": None, "tokens_in": res.get("tokens_in"),
                   "tokens_out": res.get("tokens_out"), "latency_ms": res.get("latency_ms"),
                   "provider_model": res.get("provider_model"), "error": res.get("error"),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if parsed is None and not rec["error"]:
                rec["error"] = "parse_null"
            if parsed is not None:
                try:
                    rec["prob_fail"] = float(parsed.get("prob_fail"))
                except (TypeError, ValueError):
                    rec["error"] = "no_prob_fail"
                v = parsed.get("verdict")
                rec["verdict"] = v.strip().lower() if isinstance(v, str) else None
                rec["red_flags"] = parsed.get("red_flags")
                rec["rationale"] = parsed.get("rationale")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    print(f"DONE {n} calls -> {args.out}", flush=True)


def to_csv(src: Path, dst: Path):
    """Flatten a scores jsonl to id,temperature,rep,prob_fail,verdict,error (one row per call)."""
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "temperature", "rep", "prob_fail", "verdict", "error"])
        for r in rows:
            w.writerow([r["id"], r["temperature"], r["rep"], r.get("prob_fail"), r.get("verdict"), r.get("error")])
    print(f"wrote {dst} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="CSV with columns id, description")
    ap.add_argument("--out", type=Path, help="output .jsonl (appended; resumable)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="blind_direct_v1", help="blind_direct_v1 | enhanced_v1 | path to a template with {description}")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED, help="request seed (seed + rep per repetition)")
    ap.add_argument("--no-seed", action="store_true", help="omit the seed (used for the temperature-0.7 stress test)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per description")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--reasoning", action="store_true", help="enable reasoning tokens (paper: disabled)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--env", type=Path, default=Path(".env"), help=".env file with OPENROUTER_API_KEY")
    ap.add_argument("--to-csv", nargs=2, metavar=("JSONL", "CSV"), help="flatten a scores jsonl to CSV and exit")
    args = ap.parse_args()
    if args.to_csv:
        to_csv(Path(args.to_csv[0]), Path(args.to_csv[1]))
        return
    if not args.input or not args.out:
        ap.error("--input and --out are required")
    run(args)


if __name__ == "__main__":
    main()
