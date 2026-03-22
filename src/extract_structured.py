"""
Day 5 — Structured extraction from clinical notes with confidence scores.

Supports two extractors:
  deepseek  — DeepSeek API via OpenAI-compatible endpoint
  qwen      — local Qwen3.5:9b via Ollama native API (think=false)

Usage:
    export DEEPSEEK_API_KEY=sk-...
    uv run python src/extract_structured.py --model deepseek           # DeepSeek only
    uv run python src/extract_structured.py --model qwen               # Qwen only
    uv run python src/extract_structured.py --model all                # both sequentially
    uv run python src/extract_structured.py --model all --sample 10   # quick test
"""

import argparse
import asyncio
import json
import re
import os
import time
from pathlib import Path

import httpx
import yaml
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

CFG_PATH    = Path("config/api_config.yaml")
PROMPT_PATH = Path("prompts/extraction_prompt.txt")
NOTES_PATH  = Path("data/notes/generated_notes.jsonl")

with open(CFG_PATH) as f:
    FULL_CFG = yaml.safe_load(f)

EXTRACTION_CFG = FULL_CFG["extraction"]

with open(PROMPT_PATH) as f:
    EXTRACTION_TEMPLATE = f.read()

FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs",
          "restecg", "thalach", "exang", "oldpeak", "slope"]

SYSTEM_PROMPT = ("You are a clinical data extraction specialist. "
                 "Return only valid JSON, no markdown fences.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def out_path(model_key: str) -> Path:
    p = Path(f"data/extracted/{model_key}/extracted_structured.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def parse_json_response(raw: str) -> dict | None:
    """Strict JSON parse, then regex fallback to extract first {...} block."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None



# ── DeepSeek extractor (OpenAI-compatible) ────────────────────────────────────

def _make_deepseek_client(cfg: dict) -> AsyncOpenAI:
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"Set environment variable {cfg['api_key_env']}")
    return AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"])


async def _extract_one_deepseek(client: AsyncOpenAI, cfg: dict,
                                 record: dict, sem: asyncio.Semaphore) -> dict:
    note = record.get("note")
    if not note:
        return {**record, "extraction": None, "parse_error": "no note"}

    @retry(stop=stop_after_attempt(cfg["retry_attempts"]),
           wait=wait_fixed(cfg["retry_wait_seconds"]))
    async def _call():
        async with sem:
            resp = await client.chat.completions.create(
                model=cfg["model"],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": EXTRACTION_TEMPLATE.replace("{note}", note)},
                ],
            )
        return resp

    resp = await _call()
    raw = resp.choices[0].message.content.strip()
    parsed = parse_json_response(raw)
    return {
        **record,
        "extraction": parsed,
        "extraction_tokens": resp.usage.total_tokens,
        "extractor": "deepseek",
        "parse_error": None if parsed else f"JSON parse failed: {raw[:120]}",
    }


async def run_deepseek(records: list[dict], sink) -> list[dict]:
    cfg    = EXTRACTION_CFG["deepseek"]
    client = _make_deepseek_client(cfg)
    sem    = asyncio.Semaphore(cfg["concurrency"])
    tasks  = [_extract_one_deepseek(client, cfg, rec, sem) for rec in records]
    return await _gather_with_progress(tasks, label="DeepSeek", sink=sink)


# ── Qwen extractor (Ollama native API, think=false) ───────────────────────────

async def _extract_one_qwen(cfg: dict, record: dict,
                             sem: asyncio.Semaphore,
                             client: httpx.AsyncClient) -> dict:
    note = record.get("note")
    if not note:
        return {**record, "extraction": None, "parse_error": "no note"}

    payload = {
        "model":   cfg["model"],
        "think":   cfg.get("think", False),
        "stream":  False,
        "options": {"temperature": cfg["temperature"],
                    "num_predict": cfg["max_tokens"]},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": EXTRACTION_TEMPLATE.replace("{note}", note)},
        ],
    }

    @retry(stop=stop_after_attempt(cfg["retry_attempts"]),
           wait=wait_fixed(cfg["retry_wait_seconds"]))
    async def _call():
        async with sem:
            r = await client.post(
                f"{cfg['base_url']}/api/chat",
                json=payload,
                timeout=120.0,
            )
            r.raise_for_status()
            return r.json()

    data = await _call()
    raw = data["message"]["content"].strip()
    total_tokens = (data.get("prompt_eval_count", 0) +
                    data.get("eval_count", 0))
    parsed = parse_json_response(raw)
    return {
        **record,
        "extraction": parsed,
        "extraction_tokens": total_tokens,
        "extractor": cfg["model"],
        "parse_error": None if parsed else f"JSON parse failed: {raw[:120]}",
    }


async def run_qwen(records: list[dict], sink, model_key: str) -> list[dict]:
    cfg = EXTRACTION_CFG[model_key]
    sem = asyncio.Semaphore(cfg["concurrency"])
    async with httpx.AsyncClient() as client:
        tasks = [_extract_one_qwen(cfg, rec, sem, client) for rec in records]
        return await _gather_with_progress(tasks, label=model_key, sink=sink)


# ── Shared progress wrapper (with incremental save) ───────────────────────────

async def _gather_with_progress(tasks, label: str, sink) -> list[dict]:
    results   = []
    completed = 0
    start     = time.time()
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
        except Exception as e:
            result = {"extraction": None, "parse_error": str(e)}
        results.append(result)
        sink.write(json.dumps(result, ensure_ascii=False) + "\n")
        sink.flush()
        completed += 1
        if completed % 10 == 0:
            elapsed = time.time() - start
            rate = completed / elapsed
            remaining = (len(tasks) - completed) / rate
            print(f"  [{label}] {completed}/{len(tasks)}  "
                  f"{elapsed:.0f}s elapsed  ~{remaining/60:.1f}min left")
    return results


# ── Load already-done sample_ids (for resume) ────────────────────────────────

def load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    return done


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    OLLAMA_MODELS = [k for k, v in EXTRACTION_CFG.items()
                     if v.get("provider") == "ollama"]
    ALL_MODELS    = ["deepseek"] + OLLAMA_MODELS

    parser.add_argument("--model", choices=ALL_MODELS + ["all"],
                        default="all",
                        help=f"Extractor to run. Options: {ALL_MODELS + ['all']}")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N records (for testing)")
    args = parser.parse_args()

    with open(NOTES_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if r.get("note")]

    if args.sample:
        records = records[:args.sample]
        print(f"Test mode: {args.sample} records")
    else:
        print(f"Full run: {len(records)} records")

    models = ALL_MODELS if args.model == "all" else [args.model]

    for model_key in models:
        path = out_path(model_key)

        # Resume: skip already-completed records
        done_ids = load_done_ids(path)
        todo = [r for r in records if r["sample_id"] not in done_ids]
        if done_ids:
            print(f"\n=== Resuming {model_key}: {len(done_ids)} done, "
                  f"{len(todo)} remaining ===")
        else:
            print(f"\n=== Extracting with {model_key} ({len(todo)} records) ===")

        if not todo:
            print("  Nothing to do.")
            continue

        with open(path, "a") as sink:
            if model_key == "deepseek":
                asyncio.run(run_deepseek(todo, sink))
            else:
                asyncio.run(run_qwen(todo, sink, model_key))

        # Final summary
        all_done = load_done_ids(path)
        success = sum(
            1 for line in open(path)
            if json.loads(line).get("extraction") is not None
        )
        print(f"  Total saved: {len(all_done)}  |  Success: {success}"
              f" ({success/len(all_done):.1%})")


if __name__ == "__main__":
    main()
