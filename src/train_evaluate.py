"""
Day 8-9 — Training and evaluation across all noise conditions.

Compares XGBoost / Logistic Regression / MLP / TabNet on:
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

import argparse as _ap
_te_parser = _ap.ArgumentParser()
_te_parser.add_argument("--model", default="deepseek",
                         help="Extractor key, e.g. deepseek / qwen2b / qwen9b")
_te_args, _ = _te_parser.parse_known_args()

GT_PATH   = Path("data/processed/heart_processed.csv")
NOISY_DIR = Path(f"data/noisy/{_te_args.model}")
FIG_DIR   = Path(f"figures/{_te_args.model}")
RES_DIR   = Path(f"results/{_te_args.model}")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["age","sex","cp","trestbps","chol","fbs",
          "restecg","thalach","exang","oldpeak","slope"]
N_SPLITS = 5
SEED     = 42


# ── Model definitions ─────────────────────────────────────────────────────────
def make_sklearn_models() -> dict:
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
                                   max_iter=2000, random_state=SEED)),
        ]),
    }


# ── Cross-validated evaluation ────────────────────────────────────────────────
def evaluate(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs, f1s, accs = [], [], []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

        m = make_sklearn_models()[model_name]
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, proba))
        f1s.append(f1_score(y_val, pred))
        accs.append(accuracy_score(y_val, pred))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "f1_mean":  float(np.mean(f1s)),
        "acc_mean": float(np.mean(accs)),
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
    model_names = list(make_sklearn_models().keys())
    rows = []

    total = len(datasets) * len(model_names)
    done  = 0
    for dataset_name, (X, y) in datasets.items():
        for model_name in model_names:
            metrics = evaluate(X, y, model_name)
            rows.append({
                "dataset": dataset_name,
                "model":   model_name,
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
    colors = {"XGBoost": "#e74c3c", "LogReg": "#3498db", "MLP": "#2ecc71", "TabNet": "#9b59b6"}

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


def plot_roc_comparison(datasets: dict):
    """ROC curves: GT vs LLM-extracted (XGBoost only)."""
    keys    = ["original_gt", "extracted_clean"]
    labels  = ["Ground Truth (upper bound)", f"LLM Extracted ({_te_args.model})"]
    colors  = ["#2c3e50", "#e74c3c"]
    styles  = ["-", "--"]

    fig, ax = plt.subplots(figsize=(7, 6))

    for key, label, color, ls in zip(keys, labels, colors, styles):
        if key not in datasets:
            continue
        X, y = datasets[key]
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 200)

        for train_idx, val_idx in kf.split(X, y):
            m = make_sklearn_models()["XGBoost"]
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = m.predict_proba(X.iloc[val_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y.iloc[val_idx], proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc_score(y.iloc[val_idx], proba))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_tpr  = np.std(tprs, axis=0)

        ax.plot(mean_fpr, mean_tpr, color=color, ls=ls, linewidth=2,
                label=f"{label} (AUC={mean_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.12, color=color)

    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve: Ground Truth vs LLM-Extracted Data\n(XGBoost, 5-fold CV)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_roc_comparison.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_roc_comparison.pdf")
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_roc_comparison.png")


def plot_feature_importance(datasets: dict):
    """XGBoost feature importance trained on GT data."""
    if "original_gt" not in datasets:
        return
    X, y = datasets["original_gt"]

    model = make_sklearn_models()["XGBoost"]
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=FIELDS)
    importance = importance.sort_values()

    FIELD_LABELS = {
        "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
        "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting BS",
        "restecg": "Resting ECG", "thalach": "Max Heart Rate",
        "exang": "Exercise Angina", "oldpeak": "ST Depression",
        "slope": "ST Slope",
    }
    importance.index = [FIELD_LABELS.get(f, f) for f in importance.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(importance.index, importance.values,
                   color=["#e74c3c" if v >= importance.median() else "#95a5a6"
                          for v in importance.values])
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance\n(trained on ground-truth data)", fontsize=12)
    ax.axvline(importance.median(), color="gray", linestyle="--",
               alpha=0.6, label="Median")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_importance.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_feature_importance.pdf")
    plt.close()
    print(f"Saved: {FIG_DIR}/fig_feature_importance.png")


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

    datasets = load_all_datasets()
    plot_noise_curve(results)
    plot_model_comparison(results)
    plot_roc_comparison(datasets)
    plot_feature_importance(datasets)


if __name__ == "__main__":
    main()
