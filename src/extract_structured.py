"""
Day 5 — Structured extraction from clinical notes with confidence scores.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    uv run python src/extract_structured.py               # full run
    uv run python src/extract_structured.py --sample 10   # test
"""

import argparse
import asyncio
import json
import re
import os
import time
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

CFG_PATH    = Path("config/api_config.yaml")
PROMPT_PATH = Path("prompts/extraction_prompt.txt")
NOTES_PATH  = Path("data/notes/generated_notes.jsonl")
OUT_PATH    = Path("data/extracted/extracted_structured.jsonl")

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)["extraction"]

with open(PROMPT_PATH) as f:
    EXTRACTION_TEMPLATE = f.read()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = ["age","sex","cp","trestbps","chol","fbs",
          "restecg","thalach","exang","oldpeak","slope"]


def make_client() -> AsyncOpenAI:
    api_key = os.environ.get(CFG["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"Set environment variable {CFG['api_key_env']}")
    return AsyncOpenAI(api_key=api_key, base_url=CFG["base_url"])


def parse_json_response(raw: str) -> dict | None:
    """Try strict JSON parse, then regex fallback to extract JSON block."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first {...} block
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


@retry(stop=stop_after_attempt(CFG["retry_attempts"]),
       wait=wait_fixed(CFG["retry_wait_seconds"]))
async def extract_one(client: AsyncOpenAI, record: dict,
                       sem: asyncio.Semaphore) -> dict:
    note = record["note"]
    if not note:
        return {**record, "extraction": None, "parse_error": "no note"}

    user_prompt = EXTRACTION_TEMPLATE.format(note=note)

    async with sem:
        resp = await client.chat.completions.create(
            model       = CFG["model"],
            max_tokens  = CFG["max_tokens"],
            temperature = CFG["temperature"],
            messages=[
                {"role": "system", "content":
                    "You are a clinical data extraction specialist. "
                    "Return only valid JSON, no markdown."},
                {"role": "user", "content": user_prompt},
            ],
        )

    raw = resp.choices[0].message.content.strip()
    parsed = parse_json_response(raw)

    return {
        **record,
        "extraction": parsed,
        "extraction_tokens": resp.usage.total_tokens,
        "parse_error": None if parsed else f"JSON parse failed: {raw[:100]}",
    }


async def extract_all(records: list[dict]) -> list[dict]:
    client = make_client()
    sem    = asyncio.Semaphore(CFG["concurrency"])
    tasks  = [extract_one(client, rec, sem) for rec in records]

    results = []
    completed = 0
    start = time.time()

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
        except Exception as e:
            result = {"extraction": None, "parse_error": str(e)}
        results.append(result)
        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start
            print(f"  {completed}/{len(tasks)} extracted ({elapsed:.0f}s)")

    return results


def save_results(results: list[dict], out_path: Path):
    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    success = sum(1 for r in results if r.get("extraction") is not None)
    print(f"\nSaved {len(results)} records → {out_path}")
    print(f"Extraction success rate: {success}/{len(results)} ({success/len(results):.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    with open(NOTES_PATH) as f:
        records = [json.loads(line) for line in f]

    # Only process records with a valid note
    records = [r for r in records if r.get("note")]
    if args.sample:
        records = records[:args.sample]
        print(f"Test mode: extracting {args.sample} records")
    else:
        print(f"Full run: extracting {len(records)} records")

    results = asyncio.run(extract_all(records))
    save_results(results, OUT_PATH)


if __name__ == "__main__":
    main()
