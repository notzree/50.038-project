import os

# Large pandas merges can trigger rare OpenBLAS/MKL segfaults when many threads
# compete. Set before importing NumPy/Pandas (same idea as extract_features overnight).
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from build_dataset import build_labels_and_train_table
from train_model import train_and_evaluate


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    x = (value - min_v) / (max_v - min_v)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _norm_inputs(row: pd.Series) -> dict[str, float]:
    tempo_n = _norm(_safe_float(row.get("tempo_bpm"), 0.0), 60.0, 180.0)
    rms_n = _norm(_safe_float(row.get("rms_mean"), 0.0), 0.02, 0.25)
    centroid_n = _norm(
        _safe_float(row.get("spectral_centroid_mean"), 0.0), 500.0, 5000.0
    )
    rolloff_n = _norm(
        _safe_float(row.get("spectral_rolloff_mean"), 0.0), 1500.0, 7000.0
    )
    zcr_n = _norm(_safe_float(row.get("zcr_mean"), 0.0), 0.02, 0.25)

    onset_n = _norm(_safe_float(row.get("onset_strength_mean"), 0.0), 0.5, 12.0)
    onset_std_n = _norm(_safe_float(row.get("onset_strength_std"), 0.0), 0.0, 8.0)
    tempo_stability_n = _norm(_safe_float(row.get("tempo_stability"), 0.0), 0.0, 0.35)
    stability_n = 1.0 - tempo_stability_n
    bass_n = _norm(_safe_float(row.get("bass_energy_ratio"), 0.0), 0.05, 0.45)

    mfcc1_n = _norm(_safe_float(row.get("mfcc_1_mean"), 0.0), -500.0, 100.0)
    mfcc2_n = _norm(_safe_float(row.get("mfcc_2_mean"), 0.0), -120.0, 60.0)
    mfcc3_n = _norm(_safe_float(row.get("mfcc_3_mean"), 0.0), -120.0, 120.0)
    abs_mfcc2_n = _norm(abs(_safe_float(row.get("mfcc_2_mean"), 0.0)), 0.0, 120.0)

    return {
        "tempo_n": tempo_n,
        "rms_n": rms_n,
        "centroid_n": centroid_n,
        "rolloff_n": rolloff_n,
        "zcr_n": zcr_n,
        "onset_n": onset_n,
        "onset_std_n": onset_std_n,
        "tempo_stability_n": tempo_stability_n,
        "stability_n": stability_n,
        "bass_n": bass_n,
        "mfcc1_n": mfcc1_n,
        "mfcc2_n": mfcc2_n,
        "mfcc3_n": mfcc3_n,
        "abs_mfcc2_n": abs_mfcc2_n,
    }


def _compute_formula_set(values: dict[str, float], set_id: str) -> dict[str, float]:
    t = values["tempo_n"]
    r = values["rms_n"]
    c = values["centroid_n"]
    ro = values["rolloff_n"]
    z = values["zcr_n"]
    o = values["onset_n"]
    os = values["onset_std_n"]
    s = values["stability_n"]
    ts = values["tempo_stability_n"]
    b = values["bass_n"]
    m1 = values["mfcc1_n"]
    m2 = values["mfcc2_n"]
    m3 = values["mfcc3_n"]
    am2 = values["abs_mfcc2_n"]

    if set_id == "A":
        dance = 0.35 * t + 0.35 * o + 0.30 * s
        energy = 0.45 * r + 0.30 * c + 0.25 * ro
        brightness = 0.55 * c + 0.45 * ro
        rhythm = s
        acousticness = 1.0 - (0.45 * z + 0.30 * brightness + 0.25 * energy)
        speech = 0.55 * z + 0.45 * m2
        instrumental = 0.45 * acousticness + 0.35 * b + 0.20 * (1.0 - speech)
        valence = 0.30 * t + 0.25 * brightness + 0.25 * m1 + 0.20 * m3 - 0.15 * am2
    elif set_id == "B":
        dance = 0.45 * t + 0.30 * o + 0.25 * s
        energy = 0.35 * r + 0.25 * c + 0.20 * ro + 0.20 * o
        brightness = 0.50 * c + 0.50 * ro
        rhythm = 0.70 * s + 0.30 * (1.0 - os)
        acousticness = 1.0 - (0.40 * z + 0.30 * brightness + 0.30 * energy)
        speech = 0.60 * z + 0.40 * m2
        instrumental = 0.40 * acousticness + 0.30 * b + 0.30 * (1.0 - speech)
        valence = 0.35 * t + 0.20 * brightness + 0.20 * m1 + 0.25 * m3
    elif set_id == "C":
        dance = 0.30 * t + 0.40 * o + 0.30 * s
        energy = 0.50 * r + 0.20 * c + 0.30 * ro
        brightness = 0.45 * c + 0.55 * ro
        rhythm = 0.80 * s + 0.20 * (1.0 - z)
        acousticness = 1.0 - (0.50 * z + 0.25 * brightness + 0.25 * energy)
        speech = 0.65 * z + 0.35 * m2
        instrumental = 0.55 * acousticness + 0.20 * b + 0.25 * (1.0 - speech)
        valence = 0.25 * t + 0.20 * brightness + 0.35 * m1 + 0.20 * m3
    elif set_id == "D":
        dance = ((t * o) ** 0.5) * 0.6 + s * 0.4
        energy = (r**0.5) * 0.4 + c * 0.3 + ro * 0.3
        brightness = max(c, ro)
        rhythm = 1.0 - ts
        acousticness = 1.0 - (0.40 * z + 0.30 * brightness + 0.30 * energy)
        speech = 0.60 * z + 0.40 * m2
        instrumental = 0.50 * acousticness + 0.50 * (1.0 - speech)
        valence = 0.30 * t + 0.20 * brightness + 0.30 * m1 + 0.20 * m3
    else:
        raise ValueError(f"Unknown formula set: {set_id}")

    return {
        "danceability_proxy": _clip01(dance),
        "energy_proxy": _clip01(energy),
        "acousticness_proxy": _clip01(acousticness),
        "instrumentalness_proxy": _clip01(instrumental),
        "speechiness_proxy": _clip01(speech),
        "valence_proxy": _clip01(valence),
        "brightness_proxy": _clip01(brightness),
        "rhythmic_stability_proxy": _clip01(rhythm),
        "high_level_features_version": f"formula_{set_id}",
    }


def build_high_level_features_for_set(
    features_csv: str, output_csv: str, set_id: str
) -> None:
    # Appended feature extracts can rarely contain a malformed physical line (e.g. crash
    # mid-write). Match extract_features resume behavior so optimization can proceed.
    t0 = time.monotonic()
    print(f"  Reading {features_csv} ...", flush=True)
    df = pd.read_csv(features_csv, on_bad_lines="warn", low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    n = len(df)
    print(
        f"  Formula set {set_id}: {n} tracks with status=ok -- computing proxies ...",
        flush=True,
    )
    report_every = max(5000, n // 12) if n > 8000 else max(1, n // 6)

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        vals = _norm_inputs(row)
        feats = _compute_formula_set(vals, set_id)
        rows.append({"track_id": row["track_id"], **feats})
        if i % report_every == 0 or i == n:
            pct = 100 * i // max(n, 1)
            dt = time.monotonic() - t0
            rate = i / max(dt, 1e-6)
            print(
                f"  ... high-level proxies {i}/{n} ({pct}%)  "
                f"~{rate:.0f} tracks/s  elapsed {dt:.0f}s",
                flush=True,
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df.drop_duplicates(subset=["track_id"], keep="first")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(
        f"  Wrote {output_csv} ({len(out_df)} rows) in {time.monotonic() - t0:.1f}s",
        flush=True,
    )


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_search(args) -> None:
    formula_sets = parse_csv_list(args.formula_sets)
    seeds = parse_int_list(args.seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    n_sets = len(formula_sets)
    n_seeds = len(seeds)
    t_search = time.monotonic()
    print(f"Formula sets: {formula_sets}")
    print(f"Seeds: {seeds}")
    print(f"Output dir: {output_dir}")
    print(
        f"Training speed: cv_folds={args.cv_folds}, rf_n_estimators={args.rf_n_estimators} "
        "(override with --cv-folds / --rf-n-estimators for fuller CV or a larger forest)",
        flush=True,
    )
    if getattr(args, "lr_only", False):
        print(
            "Models: logistic_regression only (--lr-only; avoids RandomForest on huge tables)",
            flush=True,
        )
    print(
        f"Progress: {n_sets} formula set(s) × "
        f"(high-level build + train table + {n_seeds} train/eval seed(s))",
        flush=True,
    )

    for si, set_id in enumerate(formula_sets, start=1):
        set_dir = output_dir / f"set_{set_id}"
        set_dir.mkdir(parents=True, exist_ok=True)

        high_level_csv = str(set_dir / "audio_features_high_level.csv")
        labels_csv = str(set_dir / "labels_appears_in_region.csv")
        train_csv = str(set_dir / "train_table.parquet")

        print(
            f"\n=== [{si}/{n_sets}] Formula set {set_id}: build high-level features "
            f"(elapsed {time.monotonic() - t_search:.0f}s) ===",
            flush=True,
        )
        build_high_level_features_for_set(args.features_csv, high_level_csv, set_id)

        print(
            f"=== [{si}/{n_sets}] Formula set {set_id}: build training table "
            f"(elapsed {time.monotonic() - t_search:.0f}s) ===",
            flush=True,
        )
        build_labels_and_train_table(
            charts_csv=args.charts_csv,
            features_csv=args.features_csv,
            track_catalog_csv=args.track_catalog_csv,
            high_level_features_csv=high_level_csv,
            labels_csv=labels_csv,
            train_csv=train_csv,
            chart_name=args.chart_name,
            genres_csv=args.genres_csv,
            nonviral_meta_csv=args.nonviral_meta_csv,
        )
        print(
            f"Finished build_labels_and_train_table for set {set_id} -> {train_csv}",
            flush=True,
        )

        for sj, seed in enumerate(seeds, start=1):
            run_dir = set_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"=== [{si}/{n_sets}] Formula set {set_id} | seed {seed} ({sj}/{n_seeds}): "
                f"train/eval (elapsed {time.monotonic() - t_search:.0f}s) ===",
                flush=True,
            )
            try:
                train_models: tuple[str, ...] = (
                    ("logistic_regression",)
                    if getattr(args, "lr_only", False)
                    else ("logistic_regression", "random_forest")
                )
                metrics = train_and_evaluate(
                    input_csv=train_csv,
                    metrics_out=str(run_dir / "model_metrics.json"),
                    model_out=str(run_dir / "model.joblib"),
                    region_metrics_out=str(run_dir / "region_metrics.csv"),
                    test_preds_out=str(run_dir / "test_predictions.csv"),
                    errors_out=str(run_dir / "error_rows.csv"),
                    test_size=args.test_size,
                    val_size=args.val_size,
                    seed=seed,
                    feature_importance_out=str(run_dir / "feature_importance.csv"),
                    metadata_out=str(run_dir / "model_metadata.json"),
                    cv_folds=args.cv_folds,
                    rf_n_estimators=args.rf_n_estimators,
                    models=train_models,
                )
            except Exception:
                print(
                    f"ERROR: train_and_evaluate failed for set {set_id} seed {seed}",
                    flush=True,
                )
                traceback.print_exc()
                raise

            selected = metrics["selected_model"]
            selected_cv = metrics["models"][selected]["cv"]
            val_best = metrics["thresholding"]["val_best_f1_threshold"]
            test_tuned = metrics["thresholding"]["test_at_tuned_threshold"]

            summary_rows.append(
                {
                    "formula_set": set_id,
                    "seed": seed,
                    "selected_model": selected,
                    "cv_f1_mean": selected_cv.get("f1_mean"),
                    "cv_roc_auc_mean": selected_cv.get("roc_auc_mean"),
                    "val_best_threshold": val_best.get("threshold"),
                    "val_best_f1": val_best.get("f1"),
                    "test_f1_tuned": test_tuned.get("f1"),
                    "test_precision_tuned": test_tuned.get("precision"),
                    "test_recall_tuned": test_tuned.get("recall"),
                    "metrics_path": str(run_dir / "model_metrics.json"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["val_best_f1", "cv_f1_mean"], ascending=[False, False]
    )

    summary_csv = output_dir / "formula_search_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    best_json = output_dir / "best_formula_result.json"
    best_json.write_text(json.dumps(best, indent=2))

    print(
        f"\n=== Formula search complete (total {time.monotonic() - t_search:.0f}s) ===",
        flush=True,
    )
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote best result: {best_json}")
    if best:
        print(json.dumps(best, indent=2))


def main() -> None:
    import faulthandler

    faulthandler.enable()

    parser = argparse.ArgumentParser(
        description="Overnight search over high-level feature formula sets"
    )
    parser.add_argument("--features-csv", default="src/data/audio_features.csv")
    parser.add_argument("--charts-csv", default="src/data/charts.csv")
    parser.add_argument("--track-catalog-csv", default="src/data/track_catalog.csv")
    parser.add_argument("--genres-csv", default=None)
    parser.add_argument(
        "--nonviral-meta-csv", default="src/data/nonviral_track_ids.csv"
    )
    parser.add_argument("--chart-name", default="top200")
    parser.add_argument(
        "--formula-sets",
        default="A,B",
        help="Comma-separated formula set ids (default: A,B)",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default="src/data/formula_search")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        metavar="K",
        help="Grouped CV folds per model during search (default 3 for speed; use 5+ for stricter CV)",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=128,
        metavar="N",
        help="RandomForest tree count during search (default 128 for speed)",
    )
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="Train only logistic regression (much lower RAM/CPU on multi-million-row tables)",
    )
    args = parser.parse_args()

    run_search(args)


if __name__ == "__main__":
    main()
