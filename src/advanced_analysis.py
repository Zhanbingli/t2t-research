"""
Advanced analysis for the T2T pipeline paper.

Sections:
  1. Paired significance test (CV fold-level AUC paired t-test)
  2. Feature importance × extraction error cross-analysis
  3. Expected Calibration Error (ECE) per model
  4. cp field confusion matrix (Qwen2b)
  5. Dual-model AUC comparison figure
  6. Noise resilience: AUC degradation summary

Usage:
    uv run python src/advanced_analysis.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
GT_PATH      = Path("data/processed/heart_processed.csv")
EXTR_PATHS   = {
    "deepseek": Path("data/extracted/deepseek/extracted_structured.jsonl"),
    "qwen2b":   Path("data/extracted/qwen2b/extracted_structured.jsonl"),
}
ACC_PATHS    = {
    "deepseek": Path("results/deepseek/extraction_accuracy.csv"),
    "qwen2b":   Path("results/qwen2b/extraction_accuracy.csv"),
}
CAL_PATHS    = {
    "deepseek": Path("results/deepseek/calibration_data.csv"),
    "qwen2b":   Path("results/qwen2b/calibration_data.csv"),
}
EXP_PATHS    = {
    "deepseek": Path("results/deepseek/experiment_results.csv"),
    "qwen2b":   Path("results/qwen2b/experiment_results.csv"),
}
FIG_DIR = Path("figures/comparison")
RES_DIR = Path("results/comparison")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope"]
SEED     = 42
N_SPLITS = 5

FIELD_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting BS",
    "restecg": "Resting ECG", "thalach": "Max HR",
    "exang": "Exercise Angina", "oldpeak": "ST Depression", "slope": "ST Slope",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_gt() -> tuple[pd.DataFrame, pd.Series]:
    gt = pd.read_csv(GT_PATH)
    X  = gt[FIELDS].fillna(gt[FIELDS].median())
    return X, gt["target"]


def load_extracted(model: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load clean extracted features (no noise)."""
    noisy_clean = Path(f"data/noisy/{model}/extracted_clean.csv")
    if noisy_clean.exists():
        df = pd.read_csv(noisy_clean).dropna(subset=FIELDS)
        return df[FIELDS], df["target"]
    return None, None


def load_raw_records(model: str) -> list[dict]:
    path = EXTR_PATHS[model]
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Section 1: Paired significance test ───────────────────────────────────────

def cv_fold_aucs(X: pd.DataFrame, y: pd.Series) -> list[float]:
    """Return per-fold AUCs from 5-fold CV using XGBoost."""
    kf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, va in kf.split(X, y):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
        m.fit(X.iloc[tr].values, y.iloc[tr].values)
        proba = m.predict_proba(X.iloc[va].values)[:, 1]
        aucs.append(roc_auc_score(y.iloc[va].values, proba))
    return aucs


def section1_significance():
    print("\n=== Section 1: Paired significance test (XGBoost, 5-fold CV) ===")
    gt_X, gt_y = load_gt()
    aucs_gt = cv_fold_aucs(gt_X, gt_y)

    rows = []
    for model in ["deepseek", "qwen2b"]:
        X, y = load_extracted(model)
        if X is None:
            continue
        aucs_ext = cv_fold_aucs(X, y)
        diff = np.array(aucs_gt) - np.array(aucs_ext)
        t, p  = stats.ttest_rel(aucs_gt, aucs_ext)
        ci_lo = np.mean(diff) - 1.96 * np.std(diff, ddof=1) / np.sqrt(N_SPLITS)
        ci_hi = np.mean(diff) + 1.96 * np.std(diff, ddof=1) / np.sqrt(N_SPLITS)
        rows.append({
            "model":         model,
            "auc_gt":        round(np.mean(aucs_gt), 4),
            "auc_extracted": round(np.mean(aucs_ext), 4),
            "auc_drop":      round(np.mean(diff), 4),
            "ci_95_lo":      round(ci_lo, 4),
            "ci_95_hi":      round(ci_hi, 4),
            "t_stat":        round(t, 3),
            "p_value":       round(p, 4),
            "significant":   p < 0.05,
        })
        print(f"  {model}: GT={np.mean(aucs_gt):.4f}  Extracted={np.mean(aucs_ext):.4f}"
              f"  drop={np.mean(diff):+.4f}  p={p:.4f}  {'*sig*' if p < 0.05 else 'ns'}")

    # DeepSeek vs Qwen2b
    X_ds, y_ds = load_extracted("deepseek")
    X_q2, y_q2 = load_extracted("qwen2b")
    if X_ds is not None and X_q2 is not None:
        aucs_ds = cv_fold_aucs(X_ds, y_ds)
        aucs_q2 = cv_fold_aucs(X_q2, y_q2)
        t, p = stats.ttest_ind(aucs_ds, aucs_q2)
        print(f"\n  DeepSeek vs Qwen2b: {np.mean(aucs_ds):.4f} vs {np.mean(aucs_q2):.4f}"
              f"  p={p:.4f}  {'*sig*' if p < 0.05 else 'ns'}")
        rows.append({
            "model":         "deepseek_vs_qwen2b",
            "auc_gt":        round(np.mean(aucs_ds), 4),
            "auc_extracted": round(np.mean(aucs_q2), 4),
            "auc_drop":      round(np.mean(aucs_ds) - np.mean(aucs_q2), 4),
            "ci_95_lo": None, "ci_95_hi": None,
            "t_stat":        round(t, 3),
            "p_value":       round(p, 4),
            "significant":   p < 0.05,
        })

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(RES_DIR / "significance_tests.csv", index=False)
    print(f"  Saved → {RES_DIR}/significance_tests.csv")
    return sig_df


# ── Section 2: Feature importance × extraction error ──────────────────────────

def section2_importance_vs_error():
    print("\n=== Section 2: Feature importance × extraction error ===")
    gt_X, gt_y = load_gt()
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    m.fit(gt_X.values, gt_y.values)
    importance = pd.Series(m.feature_importances_, index=FIELDS)

    rows = []
    for model in ["deepseek", "qwen2b"]:
        acc_df = pd.read_csv(ACC_PATHS[model], index_col="field")
        for field in FIELDS:
            rows.append({
                "model":      model,
                "field":      field,
                "label":      FIELD_LABELS[field],
                "importance": float(importance[field]),
                "error_rate": 1.0 - float(acc_df.loc[field, "accuracy"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "importance_vs_error.csv", index=False)

    # Plot: scatter importance vs error for each model
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"deepseek": "#3498db", "qwen2b": "#e74c3c"}
    labels_map = {"deepseek": "DeepSeek (commercial)", "qwen2b": "Qwen3.5-2B (local)"}

    for ax, model in zip(axes, ["deepseek", "qwen2b"]):
        sub = df[df["model"] == model]
        ax.scatter(sub["importance"], sub["error_rate"],
                   color=colors[model], s=80, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["label"], (row["importance"], row["error_rate"]),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=11)
        ax.set_ylabel("Extraction Error Rate (1 − accuracy)", fontsize=11)
        ax.set_title(labels_map[model], fontsize=12)
        ax.grid(alpha=0.3)

    fig.suptitle("Feature Importance vs. Extraction Error Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_importance_vs_error.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_importance_vs_error.pdf")
    plt.close()
    print(f"  Saved → {FIG_DIR}/fig_importance_vs_error.png")

    # Correlation
    for model in ["deepseek", "qwen2b"]:
        sub = df[df["model"] == model]
        r, p = stats.pearsonr(sub["importance"], sub["error_rate"])
        print(f"  {model}: Pearson r(importance, error_rate) = {r:.3f}  p = {p:.4f}")

    return df


# ── Section 3: Expected Calibration Error ────────────────────────────────────

def compute_ece(cal_df: pd.DataFrame) -> float:
    """ECE = weighted mean absolute deviation between confidence and accuracy."""
    n_total = cal_df["n"].sum()
    ece = (cal_df["n"] / n_total * (cal_df["actual_accuracy"] -
                                     cal_df["mean_confidence"] / 100).abs()).sum()
    return float(ece)


def section3_ece():
    print("\n=== Section 3: Expected Calibration Error (ECE) ===")
    ece_results = {}
    field_ece   = {}

    for model in ["deepseek", "qwen2b"]:
        cal_df = pd.read_csv(CAL_PATHS[model])
        ece_results[model] = compute_ece(cal_df)
        # Per-field ECE
        field_ece[model] = (cal_df.groupby("field")
                            .apply(lambda g: compute_ece(g.reset_index(drop=True)))
                            .rename(model))
        print(f"  {model}: overall ECE = {ece_results[model]:.4f}")

    # Side-by-side bar chart of per-field ECE
    ece_df = pd.concat(field_ece.values(), axis=1)
    ece_df.columns = ["deepseek", "qwen2b"]
    ece_df.index   = [FIELD_LABELS.get(f, f) for f in ece_df.index]
    ece_df = ece_df.sort_values("qwen2b", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(ece_df))
    w    = 0.35
    ax.bar(x - w/2, ece_df["deepseek"], width=w, label="DeepSeek", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, ece_df["qwen2b"],   width=w, label="Qwen3.5-2B", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ece_df.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax.set_title(
        f"Per-field ECE: DeepSeek (overall={ece_results['deepseek']:.3f})"
        f" vs Qwen3.5-2B (overall={ece_results['qwen2b']:.3f})", fontsize=11
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_ece_comparison.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_ece_comparison.pdf")
    plt.close()
    print(f"  Saved → {FIG_DIR}/fig_ece_comparison.png")

    summary = pd.DataFrame({"model": list(ece_results.keys()),
                             "overall_ECE": list(ece_results.values())})
    summary.to_csv(RES_DIR / "ece_summary.csv", index=False)
    return ece_results


# ── Section 4: cp confusion matrix (Qwen2b) ──────────────────────────────────

def section4_cp_confusion():
    print("\n=== Section 4: cp field confusion matrix (Qwen2b) ===")
    records = load_raw_records("qwen2b")

    preds, gts = [], []
    for rec in records:
        extr = rec.get("extraction") or {}
        orig = rec.get("original") or {}
        fd   = extr.get("cp", {})
        pred = fd.get("value") if isinstance(fd, dict) else None
        gt   = orig.get("cp")
        if pred is not None and gt is not None:
            try:
                preds.append(int(round(float(pred))))
                gts.append(int(round(float(gt))))
            except (ValueError, TypeError):
                pass

    classes   = sorted(set(gts + preds))
    cm        = confusion_matrix(gts, preds, labels=classes)
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Also compute for DeepSeek for comparison
    records_ds = load_raw_records("deepseek")
    preds_ds, gts_ds = [], []
    for rec in records_ds:
        extr = rec.get("extraction") or {}
        orig = rec.get("original") or {}
        fd   = extr.get("cp", {})
        pred = fd.get("value") if isinstance(fd, dict) else None
        gt   = orig.get("cp")
        if pred is not None and gt is not None:
            try:
                preds_ds.append(int(round(float(pred))))
                gts_ds.append(int(round(float(gt))))
            except (ValueError, TypeError):
                pass
    classes_ds = sorted(set(gts_ds + preds_ds))
    cm_ds      = confusion_matrix(gts_ds, preds_ds, labels=classes_ds)
    cm_ds_norm = cm_ds.astype(float) / cm_ds.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cp_labels = [f"Type {c}" for c in classes]

    for ax, cm_norm_plot, title, cl in [
        (axes[0], cm_ds_norm, "DeepSeek", classes_ds),
        (axes[1], cm_norm,    "Qwen3.5-2B", classes),
    ]:
        tick_labels = [f"Type {c}" for c in cl]
        sns.heatmap(cm_norm_plot, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Proportion"})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Ground Truth", fontsize=11)
        ax.set_title(f"cp Extraction Confusion Matrix\n{title}", fontsize=11)

    plt.suptitle("Chest Pain Type (cp) Extraction: Ground Truth vs Predicted",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_cp_confusion.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_cp_confusion.pdf")
    plt.close()
    print(f"  Saved → {FIG_DIR}/fig_cp_confusion.png")

    # Print dominant errors for Qwen2b
    cm_df = pd.DataFrame(cm, index=[f"GT={c}" for c in classes],
                         columns=[f"Pred={c}" for c in classes])
    print("\n  Qwen2b cp confusion matrix (raw counts):")
    print(cm_df.to_string())
    return cm_df


# ── Section 5: Dual-model AUC comparison ─────────────────────────────────────

def section5_dual_model_comparison():
    print("\n=== Section 5: Dual-model AUC comparison ===")
    dfs = {}
    for model in ["deepseek", "qwen2b"]:
        df = pd.read_csv(EXP_PATHS[model])
        df["extractor"] = model
        dfs[model] = df

    # Key conditions to compare
    conditions = {
        "original_gt":       "Ground Truth\n(upper bound)",
        "extracted_clean":   "Extracted\n(0% noise)",
        "noise_10pct_random": "10% Noise\n(random)",
        "noise_20pct_random": "20% Noise\n(random)",
        "noise_30pct_random": "30% Noise\n(random)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    model_clf_names = ["XGBoost", "LogReg", "MLP"]
    colors = {"deepseek": "#3498db", "qwen2b": "#e74c3c"}
    labels = {"deepseek": "DeepSeek", "qwen2b": "Qwen3.5-2B"}
    x = np.arange(len(conditions))
    w = 0.35

    for ax, clf in zip(axes, model_clf_names):
        for i, (model, df) in enumerate(dfs.items()):
            sub = df[df["model"] == clf]
            auc_vals = [sub[sub["dataset"] == cond]["auc_mean"].values[0]
                        if len(sub[sub["dataset"] == cond]) > 0 else np.nan
                        for cond in conditions]
            auc_stds = [sub[sub["dataset"] == cond]["auc_std"].values[0]
                        if len(sub[sub["dataset"] == cond]) > 0 else 0
                        for cond in conditions]
            offset = (i - 0.5) * w
            ax.bar(x + offset, auc_vals, width=w, yerr=auc_stds,
                   label=labels[model], color=colors[model],
                   alpha=0.85, capsize=4, error_kw={"linewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels(list(conditions.values()), fontsize=8.5)
        ax.set_title(clf, fontsize=12, fontweight="bold")
        ax.set_ylabel("AUC-ROC" if ax is axes[0] else "", fontsize=11)
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("DeepSeek vs Qwen3.5-2B: AUC-ROC Across Noise Conditions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_dual_model_comparison.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_dual_model_comparison.pdf")
    plt.close()
    print(f"  Saved → {FIG_DIR}/fig_dual_model_comparison.png")


# ── Section 6: Noise resilience summary ──────────────────────────────────────

def section6_noise_resilience():
    print("\n=== Section 6: Noise resilience summary ===")

    noise_map = {
        "extracted_clean":    0.0,
        "noise_05pct_random": 0.05,
        "noise_10pct_random": 0.10,
        "noise_20pct_random": 0.20,
        "noise_30pct_random": 0.30,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = {"deepseek": "#3498db", "qwen2b": "#e74c3c"}
    styles  = {"XGBoost": "-o", "LogReg": "--s", "MLP": ":^"}

    rows = []
    for model in ["deepseek", "qwen2b"]:
        df  = pd.read_csv(EXP_PATHS[model])
        label_prefix = "DeepSeek" if model == "deepseek" else "Qwen3.5-2B"

        for clf in ["XGBoost", "LogReg", "MLP"]:
            sub = df[df["model"] == clf]
            auc_clean = sub[sub["dataset"] == "extracted_clean"]["auc_mean"].values
            if len(auc_clean) == 0:
                continue
            auc_clean = float(auc_clean[0])

            xs, ys, errs = [], [], []
            for cond, lvl in noise_map.items():
                row_ = sub[sub["dataset"] == cond]
                if len(row_) == 0:
                    continue
                auc = float(row_["auc_mean"].values[0])
                xs.append(lvl)
                ys.append(auc)
                errs.append(float(row_["auc_std"].values[0]))
                rows.append({
                    "extractor": model, "classifier": clf,
                    "noise_level": lvl, "auc_mean": auc,
                    "auc_drop_from_clean": round(auc_clean - auc, 4),
                })

            fmt = styles[clf]
            ax.errorbar(xs, ys, yerr=errs, fmt=fmt,
                        color=colors[model], alpha=0.8, capsize=3,
                        label=f"{label_prefix}/{clf}", linewidth=1.8)

    ax.set_xlabel("Noise level (random strategy)", fontsize=12)
    ax.set_ylabel("AUC-ROC (5-fold CV)", fontsize=12)
    ax.set_title("Noise Resilience: AUC vs. Noise Level\n(DeepSeek vs Qwen3.5-2B, all classifiers)",
                 fontsize=12)
    ax.set_xticks([0, 0.05, 0.10, 0.20, 0.30])
    ax.set_xticklabels(["0%", "5%", "10%", "20%", "30%"])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_noise_resilience.png", dpi=150)
    fig.savefig(FIG_DIR / "fig_noise_resilience.pdf")
    plt.close()
    print(f"  Saved → {FIG_DIR}/fig_noise_resilience.png")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(RES_DIR / "noise_resilience.csv", index=False)

    # AUC drop summary at 30% noise
    print("\n  AUC drop at 30% random noise (from clean baseline):")
    drop30 = res_df[res_df["noise_level"] == 0.30].groupby(
        ["extractor", "classifier"])["auc_drop_from_clean"].mean().round(4)
    print(drop30.to_string())
    return res_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  T2T Advanced Analysis")
    print("=" * 60)

    section1_significance()
    section2_importance_vs_error()
    section3_ece()
    section4_cp_confusion()
    section5_dual_model_comparison()
    section6_noise_resilience()

    print("\n\nAll outputs saved to:")
    print(f"  figures → {FIG_DIR}/")
    print(f"  results → {RES_DIR}/")
    print("\nAdvanced analysis complete.")


if __name__ == "__main__":
    main()
