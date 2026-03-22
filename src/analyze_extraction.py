"""
Day 6 — Extraction accuracy analysis and confidence calibration.

Computes per-field accuracy metrics and plots calibration curve.

Usage:
    uv run python src/analyze_extraction.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

EXTR_PATH = Path("data/extracted/extracted_structured.jsonl")
FIG_DIR   = Path("figures")
RES_DIR   = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

FIELDS = ["age","sex","cp","trestbps","chol","fbs",
          "restecg","thalach","exang","oldpeak","slope"]

CATEGORICAL = {"sex","cp","fbs","restecg","exang","slope"}
CONTINUOUS  = {"age","trestbps","chol","thalach","oldpeak"}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    with open(EXTR_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        extr = rec.get("extraction") or {}
        orig = rec.get("original") or {}
        row = {"sample_id": rec["sample_id"]}
        for field in FIELDS:
            fd = extr.get(field, {})
            row[f"{field}_pred"] = fd.get("value") if isinstance(fd, dict) else None
            row[f"{field}_conf"] = fd.get("confidence", 0) if isinstance(fd, dict) else 0
            row[f"{field}_gt"]   = orig.get(field)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Per-field accuracy ────────────────────────────────────────────────────────
def field_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for field in FIELDS:
        pred = df[f"{field}_pred"]
        gt   = df[f"{field}_gt"]
        mask = pred.notna() & gt.notna()
        n    = mask.sum()

        if field in CATEGORICAL:
            correct = ((pred[mask].astype(float).round() ==
                        gt[mask].astype(float)).sum())
            acc = correct / n if n else 0
            stats.append({
                "field": field, "type": "categorical",
                "n": n, "accuracy": round(acc, 3),
                "mae": None, "extraction_rate": round(n / len(df), 3),
            })
        else:
            mae = mean_absolute_error(gt[mask].astype(float),
                                       pred[mask].astype(float)) if n else None
            exact = ((pred[mask].astype(float).round() ==
                      gt[mask].astype(float).round()).sum())
            acc = exact / n if n else 0
            stats.append({
                "field": field, "type": "continuous",
                "n": n, "accuracy": round(acc, 3),
                "mae": round(mae, 2) if mae else None,
                "extraction_rate": round(n / len(df), 3),
            })

    return pd.DataFrame(stats).set_index("field")


# ── Calibration analysis ──────────────────────────────────────────────────────
def calibration_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each confidence bucket, compute actual accuracy.
    Returns a DataFrame used to plot the calibration curve.
    """
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    rows = []

    for field in FIELDS:
        pred = df[f"{field}_pred"]
        gt   = df[f"{field}_gt"]
        conf = df[f"{field}_conf"]
        mask = pred.notna() & gt.notna()

        if mask.sum() < 5:
            continue

        conf_binned = pd.cut(conf[mask], bins=bins, labels=labels, right=True)
        correct = (pred[mask].astype(float).round() ==
                   gt[mask].astype(float).round()).astype(int)

        for bucket in labels:
            idx = conf_binned == bucket
            if idx.sum() < 3:
                continue
            rows.append({
                "field": field,
                "conf_bucket": bucket,
                "n": int(idx.sum()),
                "actual_accuracy": float(correct[idx].mean()),
                "mean_confidence": float(conf[mask][idx].mean()),
            })

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_accuracy_heatmap(acc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 7))
    heat = acc_df[["accuracy"]].T
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_title("Per-field extraction accuracy")
    ax.set_xlabel("Field")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_extraction_accuracy_heatmap.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_extraction_accuracy_heatmap.pdf")
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_extraction_accuracy_heatmap.png")


def plot_calibration_curve(cal_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Aggregate across all fields
    agg = cal_df.groupby("conf_bucket").agg(
        actual_accuracy=("actual_accuracy", "mean"),
        mean_confidence=("mean_confidence", "mean"),
        n=("n", "sum"),
    ).reset_index()

    ax.plot(agg["mean_confidence"], agg["actual_accuracy"],
            marker="o", linewidth=2, color="#e74c3c", label="Observed")
    ax.plot([0, 100], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.fill_between(agg["mean_confidence"],
                    agg["actual_accuracy"] - 0.05,
                    agg["actual_accuracy"] + 0.05,
                    alpha=0.15, color="#e74c3c")

    ax.set_xlabel("LLM confidence score (self-reported)", fontsize=12)
    ax.set_ylabel("Actual extraction accuracy", fontsize=12)
    ax.set_title("Confidence calibration of LLM extraction", fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_calibration_curve.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_calibration_curve.pdf")
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_calibration_curve.png")


def plot_extraction_rate_bar(acc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if r < 0.9 else "#2ecc71"
              for r in acc_df["extraction_rate"]]
    ax.bar(acc_df.index, acc_df["extraction_rate"], color=colors)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    ax.set_xlabel("Field")
    ax.set_ylabel("Extraction rate (non-null)")
    ax.set_title("Field-level extraction coverage")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_extraction_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_extraction_rate.png")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    records = load_records()
    print(f"Loaded {len(records)} records")

    df = flatten(records)
    valid = df.dropna(subset=[f"{f}_gt" for f in FIELDS], how="all")
    print(f"Records with ground truth: {len(valid)}")

    # Per-field accuracy
    acc_df = field_accuracy(valid)
    print("\n=== Per-field extraction accuracy ===")
    print(acc_df.to_string())
    acc_df.to_csv(RES_DIR / "extraction_accuracy.csv")

    # Calibration
    cal_df = calibration_analysis(valid)
    if not cal_df.empty:
        cal_df.to_csv(RES_DIR / "calibration_data.csv", index=False)
        overall_auc = cal_df.groupby("conf_bucket")["actual_accuracy"].mean()
        print("\n=== Calibration by confidence bucket ===")
        print(overall_auc.to_string())

    # Plots
    plot_accuracy_heatmap(acc_df)
    if not cal_df.empty:
        plot_calibration_curve(cal_df)
    plot_extraction_rate_bar(acc_df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
