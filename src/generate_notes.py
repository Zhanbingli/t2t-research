"""
Day 3 — Async clinical note generation via DeepSeek API.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    uv run python src/generate_notes.py               # full run (918 notes)
    uv run python src/generate_notes.py --sample 10   # quick smoke-test
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH    = Path("config/api_config.yaml")
PROMPT_PATH = Path("prompts/generation_prompt.txt")
DATA_PATH   = Path("data/processed/heart_processed.csv")
OUT_PATH    = Path("data/notes/generated_notes.jsonl")

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)["generation"]

with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = f.read()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── API client ────────────────────────────────────────────────────────────────
def make_client() -> AsyncOpenAI:
    api_key = os.environ.get(CFG["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"Set environment variable {CFG['api_key_env']}")
    return AsyncOpenAI(api_key=api_key, base_url=CFG["base_url"])


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        age       = int(row["age"]),
        sex_label = row["sex_label"],
        cp_label  = row["cp_label"],
        trestbps  = int(row["trestbps"]),
        chol      = int(row["chol"]),
        fbs_label = row["fbs_label"],
        ecg_label = row["ecg_label"],
        thalach   = int(row["thalach"]),
        exang_label = row["exang_label"],
        oldpeak   = float(row["oldpeak"]),
        slope_label = row["slope_label"],
    )


# ── Single note generation (with retry) ──────────────────────────────────────
@retry(stop=stop_after_attempt(CFG["retry_attempts"]),
       wait=wait_fixed(CFG["retry_wait_seconds"]))
async def generate_one(client: AsyncOpenAI, sample_id: int,
                        user_prompt: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        resp = await client.chat.completions.create(
            model       = CFG["model"],
            max_tokens  = CFG["max_tokens"],
            temperature = CFG["temperature"],
            messages=[
                {"role": "system", "content": (
                    "You are a senior cardiologist writing realistic clinic notes. "
                    "Follow all instructions in the user message exactly."
                )},
                {"role": "user", "content": user_prompt},
            ],
        )
    return {
        "sample_id": sample_id,
        "note": resp.choices[0].message.content.strip(),
        "tokens_used": resp.usage.total_tokens,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────
async def generate_all(df: pd.DataFrame) -> list[dict]:
    client = make_client()
    sem    = asyncio.Semaphore(CFG["concurrency"])
    tasks  = []

    for idx, row in df.iterrows():
        user_prompt = build_user_prompt(row)
        tasks.append(generate_one(client, int(idx), user_prompt, sem))

    results = []
    completed = 0
    start = time.time()

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            results.append({"sample_id": -1, "note": None, "error": str(e)})
        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start
            print(f"  {completed}/{len(tasks)} notes generated "
                  f"({elapsed:.0f}s elapsed)")

    return results


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, results: list[dict], out_path: Path):
    # Merge notes back with original structured data
    notes_by_id = {r["sample_id"]: r for r in results}

    records = []
    for idx, row in df.iterrows():
        note_data = notes_by_id.get(int(idx), {})
        record = {
            "sample_id": int(idx),
            "original": {
                "age":      int(row["age"]),
                "sex":      int(row["sex"]),
                "cp":       int(row["cp"]),
                "trestbps": int(row["trestbps"]),
                "chol":     int(row["chol"]),
                "fbs":      int(row["fbs"]),
                "restecg":  int(row["restecg"]),
                "thalach":  int(row["thalach"]),
                "exang":    int(row["exang"]),
                "oldpeak":  float(row["oldpeak"]),
                "slope":    int(row["slope"]),
                "target":   int(row["target"]),
            },
            "note": note_data.get("note"),
            "tokens_used": note_data.get("tokens_used"),
            "error": note_data.get("error"),
        }
        records.append(record)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    success = sum(1 for r in records if r["note"] is not None)
    total_tokens = sum(r["tokens_used"] or 0 for r in records)
    print(f"\nSaved {len(records)} records → {out_path}")
    print(f"Success rate: {success}/{len(records)} ({success/len(records):.1%})")
    print(f"Total tokens used: {total_tokens:,}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Generate only N notes (for testing)")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    if args.sample:
        df = df.head(args.sample)
        print(f"Smoke-test mode: generating {args.sample} notes")
    else:
        print(f"Full run: generating {len(df)} notes")

    results = asyncio.run(generate_all(df))
    save_results(df, results, OUT_PATH)


if __name__ == "__main__":
    main()
