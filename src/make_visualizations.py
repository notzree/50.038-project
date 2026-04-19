import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import learning_curve

from train_model import load_and_prepare_data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_dataset_overview(train_csv: Path, labels_csv: Path, out_path: Path) -> None:
    train_df = pd.read_csv(train_csv)
    labels_df = pd.read_csv(labels_csv)

    num_tracks = train_df["track_id"].nunique()
    num_regions = train_df["region"].nunique()
    num_pairs = len(labels_df)
    pos_count = int((labels_df["appears_in_region"] == 1).sum())
    neg_count = int((labels_df["appears_in_region"] == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: summary bars
    categories = ["Unique tracks", "Regions", "(track, region) pairs"]
    values = [num_tracks, num_regions, num_pairs]
    axes[0].bar(categories, values, color=["#5E81AC", "#88C0D0", "#81A1C1"])
    axes[0].set_title("Dataset Overview")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

    # Right: class balance donut
    total = pos_count + neg_count
    sizes = [neg_count, pos_count]
    labels = [f"Class 0 (No): {neg_count:,}", f"Class 1 (Yes): {pos_count:,}"]
    colors = ["#B0BEC5", "#E07A5F"]
    wedges, _ = axes[1].pie(
        sizes, colors=colors, startangle=90, wedgeprops={"width": 0.45}
    )
    axes[1].legend(
        wedges, labels, loc="center left", bbox_to_anchor=(0.95, 0.5), frameon=False
    )
    axes[1].set_title(f"Target Balance (positive rate: {pos_count / total:.2%})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_label_distribution(labels_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(labels_csv)
    counts = df["appears_in_region"].value_counts().sort_index()
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    total = sum(values)
    pcts = [v / total * 100 if total else 0 for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Not Appears (0)", "Appears (1)"], values, color=["#6c8ebf", "#e07a5f"]
    )
    ax.set_title("Label Distribution: appears_in_region")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    for bar, count, pct in zip(bars, values, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_distributions(train_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(train_csv)
    features = [
        "tempo_bpm",
        "rms_mean",
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
        "zcr_mean",
        "mfcc_1_mean",
    ]
    features = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    class0 = df[df["appears_in_region"] == 0]
    class1 = df[df["appears_in_region"] == 1]

    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(
            class0[feature].dropna(),
            bins=30,
            alpha=0.6,
            label="Class 0",
            color="#90A4AE",
            density=True,
        )
        ax.hist(
            class1[feature].dropna(),
            bins=30,
            alpha=0.6,
            label="Class 1",
            color="#E57373",
            density=True,
        )
        ax.set_title(feature)
        ax.tick_params(axis="x", labelrotation=20)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Feature Distributions by Label", y=1.02, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(metrics_json: Path, out_path: Path) -> None:
    with open(metrics_json) as f:
        metrics = json.load(f)

    models = metrics["models"]
    model_names = list(models.keys())
    display_names = [n.replace("_", " ").title() for n in model_names]
    metric_names = ["f1", "roc_auc", "precision", "recall"]
    colors = ["#5E81AC", "#A3BE8C", "#EBCB8B", "#BF616A", "#B48EAD"]

    x = np.arange(len(metric_names))
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_list = []
    for i, (name, display) in enumerate(zip(model_names, display_names)):
        # Support both old (flat) and new (nested with test/val/cv) metrics format
        model_data = models[name]
        if isinstance(model_data, dict) and "test" in model_data:
            model_data = model_data["test"]
        vals = [model_data.get(m, 0) for m in metric_names]
        offset = (i - (n_models - 1) / 2) * width
        b = ax.bar(
            x + offset, vals, width, label=display, color=colors[i % len(colors)]
        )
        bars_list.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bars in bars_list:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cm(ax, cm, title: str) -> None:
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    return im


def plot_confusion_matrices(metrics_json: Path, out_path: Path) -> None:
    with open(metrics_json) as f:
        metrics = json.load(f)

    # Support both old and new metrics format
    if "thresholding" in metrics:
        m = metrics["thresholding"]
        cm_default = m.get("test_at_default_0_5", m.get("val_default_0_5", {})).get(
            "confusion_matrix", [[0, 0], [0, 0]]
        )
        cm_tuned = m.get(
            "test_at_tuned_threshold", m.get("val_best_f1_threshold", {})
        ).get("confusion_matrix", [[0, 0], [0, 0]])
        t_default = 0.5
        t_tuned = m.get("val_best_f1_threshold", {}).get("threshold", 0.5)
    else:
        m = metrics["selected_model_thresholding"]
        cm_default = m["default_0_5"]["confusion_matrix"]
        cm_tuned = m["best_f1_threshold_on_test"]["confusion_matrix"]
        t_default = m["default_0_5"]["threshold"]
        t_tuned = m["best_f1_threshold_on_test"]["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_cm(axes[0], cm_default, f"Default threshold ({t_default})")
    _plot_cm(axes[1], cm_tuned, f"Tuned threshold ({t_tuned})")
    fig.suptitle("Confusion Matrices: Selected Model", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_region_performance(region_csv: Path, out_path: Path, top_n: int = 15) -> None:
    df = pd.read_csv(region_csv)
    df = (
        df.sort_values("support", ascending=False)
        .head(top_n)
        .sort_values("f1", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["region"], df["f1"], color="#81A1C1")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title(f"Per-Region F1 (Top {top_n} Regions by Support)")
    ax.grid(axis="x", alpha=0.25)

    for bar, val in zip(bars, df["f1"]):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_region_errors(error_counts_csv: Path, out_path: Path, top_n: int = 15) -> None:
    df = pd.read_csv(error_counts_csv)
    df["total_errors"] = df["false_positives"] + df["false_negatives"]
    df = df.sort_values("total_errors", ascending=False).head(top_n)
    df = df.sort_values("total_errors", ascending=True)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, df["false_positives"], color="#F2CC8F", label="False Positives")
    ax.barh(
        y,
        df["false_negatives"],
        left=df["false_positives"],
        color="#E07A5F",
        label="False Negatives",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["region"])
    ax.set_xlabel("Count")
    ax.set_title(f"Error Counts by Region (Top {top_n} by total errors)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_region_feature_lift_heatmap(
    train_csv: Path,
    out_path: Path,
    top_n_regions: int = 20,
    min_region_support: int = 100,
) -> None:
    """Heatmap of per-region proxy feature lift: mean(pos) - mean(neg)."""
    df = pd.read_csv(train_csv)
    proxy_cols = [c for c in df.columns if c.endswith("_proxy")]
    if not proxy_cols:
        print("No proxy columns found, skipping region feature lift heatmap")
        return

    support = df.groupby("region").size().rename("support")
    pos_count = df[df["appears_in_region"] == 1].groupby("region").size().rename("pos")
    neg_count = df[df["appears_in_region"] == 0].groupby("region").size().rename("neg")
    region_stats = pd.concat([support, pos_count, neg_count], axis=1).fillna(0)
    eligible = region_stats[
        (region_stats["support"] >= min_region_support)
        & (region_stats["pos"] > 0)
        & (region_stats["neg"] > 0)
    ].index

    if len(eligible) == 0:
        print("No eligible regions with both classes, skipping heatmap")
        return

    work = df[df["region"].isin(eligible)]
    pos_means = (
        work[work["appears_in_region"] == 1].groupby("region")[proxy_cols].mean()
    )
    neg_means = (
        work[work["appears_in_region"] == 0].groupby("region")[proxy_cols].mean()
    )
    lift = (pos_means - neg_means).dropna(how="all")
    if lift.empty:
        print("Lift matrix is empty, skipping heatmap")
        return

    lift["_abs_mean"] = lift.abs().mean(axis=1)
    lift = lift.sort_values("_abs_mean", ascending=False).head(top_n_regions)
    lift = lift.drop(columns=["_abs_mean"])

    fig, ax = plt.subplots(figsize=(13, 7))
    vmax = max(0.01, float(np.nanmax(np.abs(lift.values))))
    im = ax.imshow(lift.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Feature lift (mean positive - mean negative)")

    ax.set_xticks(np.arange(len(proxy_cols)))
    ax.set_xticklabels(proxy_cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(lift.index)))
    ax.set_yticklabels(lift.index)
    ax.set_title("Region-wise High-Level Feature Lift")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_region_probability_gap(
    test_preds_csv: Path,
    out_path: Path,
    top_n_regions: int = 15,
    min_region_support: int = 20,
) -> None:
    """Per-region mean predicted probability by true class (0 vs 1)."""
    df = pd.read_csv(test_preds_csv)
    required = {"region", "y_true", "y_prob"}
    if not required.issubset(df.columns):
        print("Missing region/y_true/y_prob columns in test predictions, skipping")
        return

    grouped = df.groupby(["region", "y_true"])["y_prob"].mean().reset_index()
    pivot = grouped.pivot(index="region", columns="y_true", values="y_prob").rename(
        columns={0: "neg_mean_prob", 1: "pos_mean_prob"}
    )
    support = df.groupby("region").size().rename("support")
    merged = pivot.join(support, how="inner").dropna(
        subset=["neg_mean_prob", "pos_mean_prob"]
    )
    merged = merged[merged["support"] >= min_region_support]
    if merged.empty:
        print("No eligible regions for probability gap plot, skipping")
        return

    merged["gap"] = merged["pos_mean_prob"] - merged["neg_mean_prob"]
    merged = merged.sort_values("gap", ascending=False).head(top_n_regions)
    merged = merged.sort_values("gap", ascending=True)

    y = np.arange(len(merged))
    h = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        y - h / 2, merged["neg_mean_prob"], h, color="#90A4AE", label="True class 0"
    )
    ax.barh(
        y + h / 2, merged["pos_mean_prob"], h, color="#E07A5F", label="True class 1"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(merged.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_title("Region-wise Mean Predicted Probability by True Class")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trends_coverage_heatmap(train_csv: Path, out_path: Path) -> None:
    """Visualize non-zero coverage of gt_* features by region."""
    df = pd.read_csv(train_csv)
    gt_cols = [c for c in df.columns if c.startswith("gt_")]
    if not gt_cols or "region" not in df.columns:
        print("No trends columns found; skipping trends coverage heatmap")
        return

    coverage = (
        df.groupby("region")[gt_cols]
        .apply(lambda g: (g != 0).mean())
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    if coverage.empty:
        print("Trends coverage matrix empty; skipping")
        return

    # Keep most represented regions for readability
    support = df.groupby("region").size().sort_values(ascending=False)
    regions = support.head(25).index
    coverage = coverage.loc[coverage.index.intersection(regions)]

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(coverage.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Non-zero coverage rate")

    ax.set_xticks(np.arange(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(coverage.index)))
    ax.set_yticklabels(coverage.index)
    ax.set_title("External Feature Coverage by Region")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trends_distributions_by_label(train_csv: Path, out_path: Path) -> None:
    """Show distributions of compact trends features split by label."""
    df = pd.read_csv(train_csv)
    gt_cols = [
        c
        for c in [
            "gt_region_interest",
            "gt_peak",
            "gt_mean",
            "gt_slope",
            "gt_momentum",
            "gt_weeks_above50",
        ]
        if c in df.columns
    ]
    if not gt_cols or "appears_in_region" not in df.columns:
        print("No trends columns/labels found; skipping trends distributions")
        return

    class0 = df[df["appears_in_region"] == 0]
    class1 = df[df["appears_in_region"] == 1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(gt_cols):
        ax = axes[i]
        ax.hist(
            class0[col].dropna(),
            bins=30,
            alpha=0.6,
            color="#90A4AE",
            density=True,
            label="Class 0",
        )
        ax.hist(
            class1[col].dropna(),
            bins=30,
            alpha=0.6,
            color="#E57373",
            density=True,
            label="Class 1",
        )
        ax.set_title(col)

    for j in range(len(gt_cols), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("External Feature Distributions by Label", y=1.02)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _extract_selected_test_metrics(metrics_json: Path) -> dict:
    with open(metrics_json) as f:
        metrics = json.load(f)
    selected = metrics.get("selected_model")
    if not selected:
        return {}
    model_block = metrics.get("models", {}).get(selected, {})
    if "test" in model_block:
        test_block = model_block["test"]
    else:
        # fallback for older schema
        test_block = model_block
    return {
        "selected_model": selected,
        "f1": float(test_block.get("f1", 0.0)),
        "roc_auc": float(test_block.get("roc_auc", 0.0)),
        "precision": float(test_block.get("precision", 0.0)),
        "recall": float(test_block.get("recall", 0.0)),
    }


def plot_trends_ablation_comparison(
    baseline_metrics_json: Path,
    trends_metrics_json: Path,
    out_path: Path,
) -> None:
    """Compare baseline model vs trends-enhanced model performance."""
    if not baseline_metrics_json.exists() or not trends_metrics_json.exists():
        print("Missing baseline/trends metrics for ablation chart; skipping")
        return

    base = _extract_selected_test_metrics(baseline_metrics_json)
    trend = _extract_selected_test_metrics(trends_metrics_json)
    if not base or not trend:
        print("Could not parse metrics for ablation chart; skipping")
        return

    metric_names = ["f1", "roc_auc", "precision", "recall"]
    x = np.arange(len(metric_names))
    width = 0.35

    base_vals = [base[m] for m in metric_names]
    trend_vals = [trend[m] for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(
        x - width / 2,
        base_vals,
        width,
        color="#5E81AC",
        label=f"Baseline ({base['selected_model']})",
    )
    b2 = ax.bar(
        x + width / 2,
        trend_vals,
        width,
        color="#A3BE8C",
        label=f"+Trends ({trend['selected_model']})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Ablation: Baseline vs Trends-Enhanced")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trends_error_reduction_by_region(
    baseline_error_counts_csv: Path,
    trends_error_counts_csv: Path,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Show per-region error reduction after adding trends."""
    if not baseline_error_counts_csv.exists() or not trends_error_counts_csv.exists():
        print(
            "Missing baseline/trends error count files; skipping error reduction chart"
        )
        return

    base = pd.read_csv(baseline_error_counts_csv)
    trend = pd.read_csv(trends_error_counts_csv)

    for df in (base, trend):
        df["total_errors"] = df["false_positives"] + df["false_negatives"]

    merged = (
        base[["region", "total_errors"]]
        .merge(
            trend[["region", "total_errors"]],
            on="region",
            how="outer",
            suffixes=("_baseline", "_trends"),
        )
        .fillna(0)
    )

    merged["delta_errors"] = (
        merged["total_errors_trends"] - merged["total_errors_baseline"]
    )
    merged["abs_delta"] = merged["delta_errors"].abs()
    merged = merged.sort_values("abs_delta", ascending=False).head(top_n)
    merged = merged.sort_values("delta_errors", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#A3BE8C" if d < 0 else "#E07A5F" for d in merged["delta_errors"]]
    ax.barh(merged["region"], merged["delta_errors"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Delta total errors (trends - baseline)")
    ax.set_title("Per-Region Error Change After Adding Trends")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr_curve(test_preds_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(test_preds_csv)
    required = {"y_true", "y_prob"}
    if not required.issubset(df.columns):
        print("Missing y_true/y_prob for PR curve; skipping")
        return

    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#5E81AC", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP={ap:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration_curve(
    test_preds_csv: Path, out_path: Path, n_bins: int = 10
) -> None:
    df = pd.read_csv(test_preds_csv)
    required = {"y_true", "y_prob"}
    if not required.issubset(df.columns):
        print("Missing y_true/y_prob for calibration curve; skipping")
        return

    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color="#A3BE8C", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_learning_curve(
    train_csv: Path,
    model_path: Path,
    out_path: Path,
    max_samples: int = 120000,
) -> None:
    if not model_path.exists():
        print(f"Model not found for learning curve ({model_path}); skipping")
        return

    X, y, _, _ = load_and_prepare_data(str(train_csv))
    if len(y) > max_samples:
        sample_idx = np.random.RandomState(42).choice(
            len(y), size=max_samples, replace=False
        )
        X = X.iloc[sample_idx].reset_index(drop=True)
        y = y[sample_idx]
    estimator = clone(joblib.load(model_path))

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=3,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#5E81AC", label="Train F1")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#5E81AC",
    )
    ax.plot(train_sizes, val_mean, "o-", color="#BF616A", label="Validation F1")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#BF616A"
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project visualization set")
    parser.add_argument("--train", default="src/data/train_table.csv")
    parser.add_argument("--labels", default="src/data/labels_appears_in_region.csv")
    parser.add_argument("--metrics", default="src/data/model_metrics.json")
    parser.add_argument("--region-metrics", default="src/data/region_metrics.csv")
    parser.add_argument("--error-counts", default="src/data/error_counts_by_region.csv")
    parser.add_argument("--test-preds", default="src/data/test_predictions.csv")
    parser.add_argument("--model", default="src/data/model.joblib")
    parser.add_argument("--out-dir", default="src/data/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    plot_dataset_overview(
        Path(args.train), Path(args.labels), out_dir / "01_dataset_overview.png"
    )
    plot_label_distribution(Path(args.labels), out_dir / "02_label_distribution.png")
    plot_feature_distributions(
        Path(args.train), out_dir / "03_feature_distributions.png"
    )
    plot_model_comparison(Path(args.metrics), out_dir / "04_model_comparison.png")
    plot_confusion_matrices(Path(args.metrics), out_dir / "05_confusion_matrices.png")
    plot_region_performance(
        Path(args.region_metrics), out_dir / "06_region_performance.png"
    )
    plot_region_errors(Path(args.error_counts), out_dir / "07_region_errors.png")
    plot_region_feature_lift_heatmap(
        Path(args.train), out_dir / "08_region_feature_lift_heatmap.png"
    )
    plot_region_probability_gap(
        Path(args.test_preds), out_dir / "09_region_probability_gap.png"
    )

    # Core diagnostics
    plot_pr_curve(Path(args.test_preds), out_dir / "14_pr_curve.png")
    plot_calibration_curve(Path(args.test_preds), out_dir / "15_calibration_curve.png")
    plot_learning_curve(
        Path(args.train), Path(args.model), out_dir / "16_learning_curve.png"
    )

    print(f"Wrote visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
