import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(metrics_json: Path, out_path: Path) -> None:
    with open(metrics_json) as f:
        metrics = json.load(f)

    models = metrics["models"]
    model_names = ["logistic_regression", "random_forest"]
    display_names = ["Logistic Regression", "Random Forest"]
    metric_names = ["f1", "roc_auc", "precision", "recall"]

    x = np.arange(len(metric_names))
    width = 0.35

    vals1 = [models[model_names[0]][m] for m in metric_names]
    vals2 = [models[model_names[1]][m] for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, vals1, width, label=display_names[0], color="#5E81AC")
    b2 = ax.bar(x + width / 2, vals2, width, label=display_names[1], color="#A3BE8C")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on MVP Test Split")
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
                fontsize=9,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Week 8 visualization set")
    parser.add_argument("--train", default="src/data/train_table_mvp.csv")
    parser.add_argument("--labels", default="src/data/labels_appears_in_region.csv")
    parser.add_argument("--metrics", default="src/data/mvp_metrics.json")
    parser.add_argument("--region-metrics", default="src/data/mvp_region_metrics.csv")
    parser.add_argument(
        "--error-counts", default="src/data/mvp_error_counts_by_region.csv"
    )
    parser.add_argument("--out-dir", default="src/data/plots/week8")
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

    print(f"Wrote Week 8 visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
