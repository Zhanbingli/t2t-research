"""
Day 8-9 — Training and evaluation across all noise conditions.

Compares XGBoost / Logistic Regression / MLP on:
  - original ground-truth data (upper bound)
  - clean extracted data (0% noise)
  - 5 noise levels × 2 strategies = 10 noisy variants

Usage:
    uv run python src/train_evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

GT_PATH   = Path("data/processed/heart_processed.csv")
NOISY_DIR = Path("data/noisy")
FIG_DIR   = Path("figures")
RES_DIR   = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

FIELDS = ["age","sex","cp","trestbps","chol","fbs",
          "restecg","thalach","exang","oldpeak","slope"]
N_SPLITS = 5
SEED     = 42


# ── Model definitions ─────────────────────────────────────────────────────────
def make_models() -> dict:
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        ),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=500, random_state=SEED)),
        ]),
    }


# ── Cross-validated evaluation ────────────────────────────────────────────────
def evaluate(X: pd.DataFrame, y: pd.Series,
             model_name: str, model) -> dict:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs, f1s, accs = [], [], []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = make_models()[model_name]   # fresh clone each fold
        m.fit(X_tr, y_tr)

        proba = m.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, proba))
        f1s.append(f1_score(y_val, pred))
        accs.append(accuracy_score(y_val, pred))

    return {
        "auc_mean":  float(np.mean(aucs)),
        "auc_std":   float(np.std(aucs)),
        "f1_mean":   float(np.mean(f1s)),
        "acc_mean":  float(np.mean(accs)),
    }


# ── Load all datasets ─────────────────────────────────────────────────────────
def load_all_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    datasets = {}

    # Ground truth (upper bound)
    gt = pd.read_csv(GT_PATH)
    datasets["original_gt"] = (gt[FIELDS].fillna(gt[FIELDS].median()),
                                 gt["target"])

    # Clean extracted
    clean_path = NOISY_DIR / "extracted_clean.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path).dropna(subset=FIELDS)
        datasets["extracted_clean"] = (df[FIELDS], df["target"])

    # Noisy variants
    for csv_path in sorted(NOISY_DIR.glob("noise_*.csv")):
        df = pd.read_csv(csv_path).dropna(subset=FIELDS)
        datasets[csv_path.stem] = (df[FIELDS], df["target"])

    return datasets


# ── Main experiment loop ──────────────────────────────────────────────────────
def run_experiments() -> pd.DataFrame:
    datasets = load_all_datasets()
    model_names = list(make_models().keys())
    rows = []

    total = len(datasets) * len(model_names)
    done  = 0
    for dataset_name, (X, y) in datasets.items():
        for model_name in model_names:
            metrics = evaluate(X, y, model_name, None)
            rows.append({
                "dataset":    dataset_name,
                "model":      model_name,
                **metrics,
            })
            done += 1
            print(f"[{done}/{total}] {dataset_name} | {model_name}"
                  f"  AUC={metrics['auc_mean']:.3f}±{metrics['auc_std']:.3f}")

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_noise_curve(results: pd.DataFrame):
    """AUC vs noise level for each model — the core result figure."""

    # Parse noise level from dataset name
    def parse_noise(name: str) -> float | None:
        if name == "original_gt":      return -0.05   # leftmost point
        if name == "extracted_clean":   return 0.0
        try:
            return int(name.split("_")[1].replace("pct","")) / 100
        except Exception:
            return None

    # Use only random-strategy rows (or clean/gt)
    mask = results["dataset"].apply(
        lambda n: "random" in n or n in ("original_gt", "extracted_clean")
    )
    df = results[mask].copy()
    df["noise_level"] = df["dataset"].apply(parse_noise)
    df = df.dropna(subset=["noise_level"]).sort_values("noise_level")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"XGBoost": "#e74c3c", "LogReg": "#3498db", "MLP": "#2ecc71"}

    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("noise_level")
        ax.errorbar(grp["noise_level"], grp["auc_mean"],
                    yerr=grp["auc_std"], label=model_name,
                    color=colors.get(model_name, "gray"),
                    marker="o", linewidth=2, capsize=4)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="extracted (0% noise)")
    ax.set_xlabel("Noise level injected", fontsize=12)
    ax.set_ylabel("AUC-ROC (5-fold CV)", fontsize=12)
    ax.set_title("Model fault tolerance under LLM extraction noise", fontsize=13)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.20, 0.30])
    ax.set_xticklabels(["GT", "0%", "5%", "10%", "20%", "30%"])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_performance_vs_noise.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_performance_vs_noise.pdf")
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_performance_vs_noise.png")


def plot_model_comparison(results: pd.DataFrame):
    """Bar chart: model × dataset comparison."""
    pivot = results.pivot_table(index="dataset", columns="model",
                                 values="auc_mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, rot=30, colormap="Set2")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC comparison across all conditions")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_model_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=== T2T Fault-Tolerance Experiment ===\n")
    results = run_experiments()

    out_csv = RES_DIR / "experiment_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    print("\n=== Summary (AUC mean) ===")
    pivot = results.pivot_table(index="dataset", columns="model",
                                 values="auc_mean").round(3)
    print(pivot.to_string())

    plot_noise_curve(results)
    plot_model_comparison(results)


if __name__ == "__main__":
    main()
