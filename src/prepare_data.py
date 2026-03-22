"""
Day 1 — Data preparation and EDA.
Loads heart_918.csv, maps categorical fields to numeric,
saves processed CSV and prints summary statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RAW_PATH = Path("data/raw/heart_918.csv")
OUT_PATH = Path("data/processed/heart_processed.csv")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Categorical encodings (UCI-aligned) ──────────────────────────────────────
SEX_MAP = {"M": 1, "F": 0}

CP_MAP = {
    "TA":  1,   # typical angina
    "ATA": 2,   # atypical angina
    "NAP": 3,   # non-anginal pain
    "ASY": 4,   # asymptomatic
}

ECG_MAP = {
    "Normal": 0,
    "ST":     1,   # ST-T wave abnormality
    "LVH":    2,   # left ventricular hypertrophy
}

ANGINA_MAP = {"Y": 1, "N": 0}

SLOPE_MAP = {
    "Up":   1,
    "Flat": 2,
    "Down": 3,
}

# Human-readable labels for the generation prompt
CP_LABEL = {1: "typical angina", 2: "atypical angina",
             3: "non-anginal pain", 4: "asymptomatic"}
ECG_LABEL = {0: "normal", 1: "ST-T wave abnormality",
              2: "left ventricular hypertrophy"}
SLOPE_LABEL = {1: "upsloping", 2: "flat", 3: "downsloping"}


def load_and_encode(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["sex"]    = df["Sex"].map(SEX_MAP)
    df["cp"]     = df["ChestPainType"].map(CP_MAP)
    df["restecg"] = df["RestingECG"].map(ECG_MAP)
    df["exang"]  = df["ExerciseAngina"].map(ANGINA_MAP)
    df["slope"]  = df["ST_Slope"].map(SLOPE_MAP)

    df = df.rename(columns={
        "Age":         "age",
        "RestingBP":   "trestbps",
        "Cholesterol": "chol",
        "FastingBS":   "fbs",
        "MaxHR":       "thalach",
        "Oldpeak":     "oldpeak",
        "HeartDisease": "target",
    })

    cols = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","target"]
    df = df[cols].copy()

    # Add readable labels for prompt generation
    df["cp_label"]     = df["cp"].map(CP_LABEL)
    df["ecg_label"]    = df["restecg"].map(ECG_LABEL)
    df["slope_label"]  = df["slope"].map(SLOPE_LABEL)
    df["sex_label"]    = df["sex"].map({1: "male", 0: "female"})
    df["exang_label"]  = df["exang"].map({1: "yes", 0: "no"})
    df["fbs_label"]    = df["fbs"].map({1: "elevated (>120 mg/dL)", 0: "normal"})

    return df


def print_summary(df: pd.DataFrame):
    numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]
    print("\n=== Dataset Summary ===")
    print(f"Samples: {len(df)}  |  Features: 11  |  Positive: {df['target'].sum()} ({df['target'].mean():.1%})")
    print("\nNumeric features:")
    print(df[numeric_cols].describe().round(2).to_string())
    print("\nClass distribution:")
    print(df["target"].value_counts().to_string())


def plot_eda(df: pd.DataFrame):
    numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]

    # Distribution by target
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 4))
    for ax, col in zip(axes, numeric_cols):
        for label, grp in df.groupby("target"):
            ax.hist(grp[col], bins=20, alpha=0.6, label=f"target={label}")
        ax.set_title(col)
        ax.legend(fontsize=7)
    plt.suptitle("Feature distributions by target")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols + ["target"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_correlation.png", dpi=150)
    plt.close()

    print(f"EDA figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    df = load_and_encode(RAW_PATH)
    print_summary(df)
    plot_eda(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nProcessed data saved to {OUT_PATH}  ({len(df)} rows)")
