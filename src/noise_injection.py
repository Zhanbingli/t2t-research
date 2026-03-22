"""
Day 7 — Gradient noise injection for fault-tolerance experiments.

Generates 10 noisy dataset variants:
  5 noise levels × 2 strategies (random / confidence-guided)

Usage:
    uv run python src/noise_injection.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG_PATH   = Path("config/api_config.yaml")
EXTR_PATH  = Path("data/extracted/extracted_structured.jsonl")
OUT_DIR    = Path("data/noisy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

NOISE_LEVELS = CFG["noise_levels"]
SEED         = CFG["random_seed"]

FIELDS = ["age","sex","cp","trestbps","chol","fbs",
          "restecg","thalach","exang","oldpeak","slope"]

# Field type info for realistic perturbations
INT_FIELDS   = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","slope"]
FLOAT_FIELDS = ["oldpeak"]

# Valid categorical ranges
VALID_VALUES = {
    "sex":     [0, 1],
    "cp":      [1, 2, 3, 4],
    "fbs":     [0, 1],
    "restecg": [0, 1, 2],
    "exang":   [0, 1],
    "slope":   [1, 2, 3],
}


def load_extracted() -> tuple[pd.DataFrame, list[dict]]:
    """Returns (numeric DataFrame of extracted values, raw records)."""
    records = []
    with open(EXTR_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    rows = []
    for rec in records:
        extr = rec.get("extraction") or {}
        row = {"sample_id": rec["sample_id"]}
        for field in FIELDS:
            field_data = extr.get(field, {})
            row[field] = field_data.get("value") if isinstance(field_data, dict) else None
            row[f"{field}_conf"] = (field_data.get("confidence", 0)
                                    if isinstance(field_data, dict) else 0)
        # Ground truth
        orig = rec.get("original", {})
        for field in FIELDS:
            row[f"{field}_gt"] = orig.get(field)
        row["target"] = orig.get("target")
        rows.append(row)

    return pd.DataFrame(rows), records


def compute_feature_stats(df: pd.DataFrame) -> dict:
    """Compute std of each numeric field for Gaussian noise scaling."""
    stats = {}
    for field in FIELDS:
        vals = df[field].dropna()
        stats[field] = {"std": float(vals.std()), "mean": float(vals.mean())}
    return stats


def perturb_field(field: str, value, stats: dict, rng: np.random.Generator):
    """Apply a single realistic perturbation to a field value."""
    if value is None:
        return value

    if field in VALID_VALUES:
        # Categorical: replace with a different valid value
        options = [v for v in VALID_VALUES[field] if v != value]
        return rng.choice(options) if options else value

    # Continuous: Gaussian noise scaled to feature std
    std = stats[field]["std"]
    noise = rng.normal(0, std * 0.5)

    if field in INT_FIELDS:
        return max(0, int(round(value + noise)))
    return round(float(value) + noise, 1)


def inject_noise(df: pd.DataFrame, stats: dict,
                 noise_level: float, strategy: str,
                 seed: int) -> pd.DataFrame:
    """
    noise_level : fraction of fields to corrupt per sample
    strategy    : "random" | "confidence"
                  confidence → corrupts fields with lowest confidence first
    """
    rng = np.random.default_rng(seed)
    result = df.copy()

    for idx in df.index:
        if strategy == "confidence":
            # Sort fields by confidence ascending (least confident first)
            confs = {f: df.at[idx, f"{f}_conf"] for f in FIELDS}
            ordered = sorted(FIELDS, key=lambda f: confs[f])
        else:
            ordered = FIELDS.copy()
            rng.shuffle(ordered)

        n_corrupt = max(1, int(round(len(FIELDS) * noise_level)))
        to_corrupt = ordered[:n_corrupt]

        for field in to_corrupt:
            original_val = df.at[idx, field]
            result.at[idx, field] = perturb_field(field, original_val, stats, rng)

    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns (no conf columns, no gt columns)."""
    return df[FIELDS + ["target"]].copy()


def main():
    print("Loading extracted data...")
    df, _ = load_extracted()

    valid = df.dropna(subset=FIELDS)
    print(f"Usable records after dropping NaN: {len(valid)}/{len(df)}")

    stats = compute_feature_stats(valid)

    saved = []
    for level in NOISE_LEVELS:
        for strategy in ["random", "confidence"]:
            noisy = inject_noise(valid, stats, level, strategy, SEED)
            out_df = build_feature_matrix(noisy)
            label = f"noise_{int(level*100):02d}pct_{strategy}"
            out_path = OUT_DIR / f"{label}.csv"
            out_df.to_csv(out_path, index=False)
            saved.append((label, len(out_df)))
            print(f"  Saved {label}.csv  ({len(out_df)} rows)")

    # Also save the clean extracted baseline (0% noise)
    clean = build_feature_matrix(valid)
    clean.to_csv(OUT_DIR / "extracted_clean.csv", index=False)
    print(f"  Saved extracted_clean.csv  ({len(clean)} rows)")

    print(f"\nDone. {len(saved)} noisy datasets saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
