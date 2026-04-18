import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NON_FEATURE_COLS = {"track_id", "appears_in_region"}


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    input_csv: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    """Load CSV, separate features / target / metadata."""
    print(f"Loading training table: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = {"track_id", "region", "appears_in_region"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found in input table")

    X = df[feature_cols]
    y = df["appears_in_region"].to_numpy()
    meta = df[["track_id", "region"]]

    print(
        f"Rows={len(df)}, features={len(feature_cols)}, positive_rate={float(y.mean()):.4f}"
    )
    return X, y, meta, feature_cols


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict:
    """Three-way group-aware split: train / val / test.

    Splits by track_id so that all rows for a given track land in the same
    partition.  This prevents data leakage where the model sees the same
    track's audio features in both train and test (just for different regions).
    """
    groups = meta["track_id"].values

    # First split: trainval vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    meta_trainval, meta_test = meta.iloc[trainval_idx], meta.iloc[test_idx]
    groups_trainval = groups[trainval_idx]

    # Second split: train vs val (val_size is relative to the whole dataset)
    val_fraction = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
    meta_train, meta_val = meta_trainval.iloc[train_idx], meta_trainval.iloc[val_idx]

    # Verify no track leakage
    train_tracks = set(meta_train["track_id"])
    val_tracks = set(meta_val["track_id"])
    test_tracks = set(meta_test["track_id"])
    assert train_tracks.isdisjoint(val_tracks), "track leakage: train & val overlap"
    assert train_tracks.isdisjoint(test_tracks), "track leakage: train & test overlap"
    assert val_tracks.isdisjoint(test_tracks), "track leakage: val & test overlap"

    print(
        f"Split: train={len(y_train)} ({len(train_tracks)} tracks), "
        f"val={len(y_val)} ({len(val_tracks)} tracks), "
        f"test={len(y_test)} ({len(test_tracks)} tracks)"
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    categorical_features = [
        c for c in feature_cols if c in {"region", "primary_genre", "source_type"}
    ]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# ---------------------------------------------------------------------------
# Model construction (single source of truth for all 4 model types)
# ---------------------------------------------------------------------------


def _make_model(name: str, seed: int, **params):
    """Construct a classifier by name. Override defaults with **params."""
    if name == "logistic_regression":
        defaults = dict(max_iter=2000, class_weight="balanced", random_state=seed)
        defaults.update(params)
        return LogisticRegression(**defaults)

    elif name == "random_forest":
        defaults = dict(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    else:
        raise ValueError(f"Unknown model: {name}")


def build_model_candidates(
    seed: int, y_train: np.ndarray | None = None
) -> dict[str, object]:
    """Return a dict of name -> classifier instance."""
    candidates = {
        "logistic_regression": _make_model("logistic_regression", seed),
        "random_forest": _make_model("random_forest", seed),
    }

    return candidates


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_models(
    candidates: dict[str, object],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> dict[str, dict]:
    """Run group-aware stratified k-fold CV for each candidate.

    Folds are split by track_id so the same track never appears in both the
    training and validation side of a fold.
    """
    groups = meta_train["track_id"].values
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    cv_results = {}

    for name, clf in candidates.items():
        print(f"  Cross-validating {name} ({k}-fold, grouped by track_id)...")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

        f1_scores = []
        roc_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups):
            pipe_clone = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", clf.__class__(**clf.get_params()))]
            )
            pipe_clone.fit(X_train.iloc[train_idx], y_train[train_idx])
            y_pred = pipe_clone.predict(X_train.iloc[val_idx])
            y_prob = pipe_clone.predict_proba(X_train.iloc[val_idx])[:, 1]
            f1_scores.append(float(f1_score(y_train[val_idx], y_pred, zero_division=0)))
            roc_scores.append(float(roc_auc_score(y_train[val_idx], y_prob)))

        f1_scores = np.array(f1_scores)
        roc_scores = np.array(roc_scores)

        cv_results[name] = {
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "roc_auc_mean": float(np.mean(roc_scores)),
            "roc_auc_std": float(np.std(roc_scores)),
            "f1_folds": f1_scores.tolist(),
            "roc_auc_folds": roc_scores.tolist(),
        }
        print(
            f"    F1={cv_results[name]['f1_mean']:.4f} "
            f"(+/- {cv_results[name]['f1_std']:.4f}), "
            f"ROC-AUC={cv_results[name]['roc_auc_mean']:.4f}"
        )

    return cv_results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_model(clf: Pipeline, X: pd.DataFrame, y) -> dict:
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def _evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _find_best_threshold(y_true, y_prob) -> dict:
    candidates = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = None
    for threshold in candidates:
        metrics = _evaluate_at_threshold(y_true, y_prob, threshold)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best


def _build_region_metrics(clf: Pipeline, X: pd.DataFrame, y) -> pd.DataFrame:
    results = X.copy()
    results["y_true"] = y
    results["y_pred"] = clf.predict(X)

    rows = []
    for region, grp in results.groupby("region"):
        rows.append(
            {
                "region": region,
                "support": int(len(grp)),
                "positive_rate": float(grp["y_true"].mean()),
                "accuracy": float(accuracy_score(grp["y_true"], grp["y_pred"])),
                "precision": float(
                    precision_score(grp["y_true"], grp["y_pred"], zero_division=0)
                ),
                "recall": float(
                    recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
                ),
                "f1": float(f1_score(grp["y_true"], grp["y_pred"], zero_division=0)),
            }
        )

    return pd.DataFrame(rows).sort_values(["f1", "support"], ascending=[False, False])


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def _extract_feature_importance(
    pipeline: Pipeline, model_name: str
) -> pd.DataFrame | None:
    """Extract feature importance from the trained pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        return None

    if len(importances) != len(feature_names):
        return None

    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Select, evaluate, and save
# ---------------------------------------------------------------------------


def select_and_evaluate(
    candidates: dict[str, object],
    cv_results: dict[str, dict],
    preprocessor: ColumnTransformer,
    splits: dict,
) -> tuple[str, Pipeline, dict]:
    """Select best model by CV F1, fit on train, tune threshold on val, evaluate on test."""

    best_name = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    print(f"Selected best model by CV F1: {best_name}")

    best_clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", candidates[best_name])]
    )
    best_clf.fit(splits["X_train"], splits["y_train"])

    # Evaluate all models on val set for comparison
    model_metrics = {}
    for name, clf_obj in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf_obj)])
        pipe.fit(splits["X_train"], splits["y_train"])
        model_metrics[name] = {
            "val": _evaluate_model(pipe, splits["X_val"], splits["y_val"]),
            "test": _evaluate_model(pipe, splits["X_test"], splits["y_test"]),
            "cv": cv_results.get(name, {}),
        }

    # Threshold tuning on VALIDATION set (not test)
    val_prob = best_clf.predict_proba(splits["X_val"])[:, 1]
    default_threshold_metrics = _evaluate_at_threshold(splits["y_val"], val_prob, 0.5)
    tuned_threshold_metrics = _find_best_threshold(splits["y_val"], val_prob)

    # Final test evaluation at tuned threshold
    test_prob = best_clf.predict_proba(splits["X_test"])[:, 1]
    tuned_threshold = tuned_threshold_metrics["threshold"]
    test_at_tuned = _evaluate_at_threshold(splits["y_test"], test_prob, tuned_threshold)
    test_at_default = _evaluate_at_threshold(splits["y_test"], test_prob, 0.5)

    metrics = {
        "models": model_metrics,
        "selected_model": best_name,
        "thresholding": {
            "val_default_0_5": default_threshold_metrics,
            "val_best_f1_threshold": tuned_threshold_metrics,
            "test_at_tuned_threshold": test_at_tuned,
            "test_at_default_0_5": test_at_default,
        },
    }

    return best_name, best_clf, metrics


def save_artifacts(
    best_name: str,
    best_clf: Pipeline,
    metrics: dict,
    splits: dict,
    feature_cols: list[str],
    *,
    metrics_out: str,
    model_out: str,
    region_metrics_out: str,
    test_preds_out: str,
    errors_out: str,
    feature_importance_out: str,
    metadata_out: str,
    dataset_info: dict,
    tuned_params: dict | None = None,
) -> dict:
    """Write all artifacts to disk."""

    # -- Test predictions --
    test_prob = best_clf.predict_proba(splits["X_test"])[:, 1]
    tuned_threshold = metrics["thresholding"]["val_best_f1_threshold"]["threshold"]

    pred_df = splits["meta_test"].copy()
    pred_df["y_true"] = splits["y_test"]
    pred_df["y_prob"] = test_prob
    pred_df["y_pred_default_0_5"] = (test_prob >= 0.5).astype(int)
    pred_df["y_pred_tuned"] = (test_prob >= tuned_threshold).astype(int)

    pred_path = Path(test_preds_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    # -- Error analysis --
    errors_df = pred_df.copy()
    errors_df["error_type"] = "correct"
    errors_df.loc[
        (errors_df["y_true"] == 1) & (errors_df["y_pred_default_0_5"] == 0),
        "error_type",
    ] = "false_negative"
    errors_df.loc[
        (errors_df["y_true"] == 0) & (errors_df["y_pred_default_0_5"] == 1),
        "error_type",
    ] = "false_positive"
    error_rows = errors_df[errors_df["error_type"] != "correct"]
    errors_path = Path(errors_out)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    error_rows.to_csv(errors_path, index=False)

    fp_by_region = (
        error_rows[error_rows["error_type"] == "false_positive"]
        .groupby("region")
        .size()
        .reset_index(name="false_positives")
    )
    fn_by_region = (
        error_rows[error_rows["error_type"] == "false_negative"]
        .groupby("region")
        .size()
        .reset_index(name="false_negatives")
    )
    error_by_region = fp_by_region.merge(fn_by_region, on="region", how="outer").fillna(
        0
    )
    error_by_region["false_positives"] = error_by_region["false_positives"].astype(int)
    error_by_region["false_negatives"] = error_by_region["false_negatives"].astype(int)
    error_by_region_path = errors_path.with_name("error_counts_by_region.csv")
    error_by_region.to_csv(error_by_region_path, index=False)

    # -- Region metrics --
    region_metrics_df = _build_region_metrics(
        best_clf, splits["X_test"], splits["y_test"]
    )
    region_metrics_path = Path(region_metrics_out)
    region_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    region_metrics_df.to_csv(region_metrics_path, index=False)

    # -- Feature importance --
    fi_df = _extract_feature_importance(best_clf, best_name)
    fi_path = Path(feature_importance_out)
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    if fi_df is not None:
        fi_df.to_csv(fi_path, index=False)
        print(f"Wrote feature importance to {fi_path}")
    else:
        print("Could not extract feature importance")

    # -- Model --
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, model_path)

    # -- Model metadata (for frontend) --
    X_train = splits["X_train"]
    valid_regions = (
        sorted(X_train["region"].unique().tolist())
        if "region" in X_train.columns
        else []
    )
    valid_genres = (
        sorted(X_train["primary_genre"].dropna().unique().tolist())
        if "primary_genre" in X_train.columns
        else []
    )

    test_perf = metrics["thresholding"]["test_at_tuned_threshold"]
    metadata = {
        "feature_columns": feature_cols,
        "feature_set": "full" if len(feature_cols) > 15 else "basic",
        "valid_regions": valid_regions,
        "valid_genres": valid_genres,
        "best_threshold": tuned_threshold,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_name,
        "performance_summary": {
            "f1": test_perf["f1"],
            "precision": test_perf["precision"],
            "recall": test_perf["recall"],
            "accuracy": test_perf["accuracy"],
        },
    }
    if tuned_params:
        metadata["tuned_hyperparameters"] = tuned_params

    metadata_path = Path(metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # -- Full metrics JSON --
    full_metrics = {
        "dataset": dataset_info,
        **metrics,
        "artifacts": {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "region_metrics_path": str(region_metrics_path),
            "test_predictions_path": str(pred_path),
            "error_rows_path": str(errors_path),
            "error_counts_by_region_path": str(error_by_region_path),
            "feature_importance_path": str(fi_path),
        },
    }

    out_path = Path(metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_metrics, indent=2))

    print(f"Wrote metrics to {metrics_out}")
    print(f"Wrote best model ({best_name}) to {model_out}")
    print(f"Wrote model metadata to {metadata_out}")
    print(f"Wrote region metrics to {region_metrics_out}")
    print(f"Wrote test predictions to {test_preds_out}")
    print(f"Wrote error rows to {errors_out}")
    print(f"Wrote error counts by region to {error_by_region_path}")
    print(json.dumps(full_metrics, indent=2))

    return full_metrics


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def train_and_evaluate(
    input_csv: str,
    metrics_out: str,
    model_out: str,
    region_metrics_out: str,
    test_preds_out: str,
    errors_out: str,
    test_size: float,
    seed: int,
    val_size: float = 0.2,
    feature_importance_out: str | None = None,
    metadata_out: str | None = None,
) -> dict:
    # Defaults for new output paths
    if feature_importance_out is None:
        feature_importance_out = str(
            Path(metrics_out).with_name("feature_importance.csv")
        )
    if metadata_out is None:
        metadata_out = str(Path(model_out).with_name("model_metadata.json"))

    # 1. Load data
    X, y, meta, feature_cols = load_and_prepare_data(input_csv)

    dataset_info = {
        "num_rows": len(y),
        "num_features": len(feature_cols),
        "test_size": test_size,
        "val_size": val_size,
        "seed": seed,
        "positive_rate": float(y.mean()),
    }

    # 2. Split
    splits = split_data(X, y, meta, test_size=test_size, val_size=val_size, seed=seed)

    # 3. Build preprocessor and candidates
    preprocessor = _build_preprocessor(feature_cols)
    candidates = build_model_candidates(seed, y_train=splits["y_train"])

    # 4. Cross-validate (grouped by track_id to prevent leakage)
    print("Running cross-validation...")
    cv_results = cross_validate_models(
        candidates, preprocessor, splits["X_train"], splits["y_train"],
        meta_train=splits["meta_train"], seed=seed,
    )

    # 5. Select best, evaluate on val + test
    best_name, best_clf, metrics = select_and_evaluate(
        candidates, cv_results, preprocessor, splits
    )

    # 6. Save everything
    return save_artifacts(
        best_name,
        best_clf,
        metrics,
        splits,
        feature_cols,
        metrics_out=metrics_out,
        model_out=model_out,
        region_metrics_out=region_metrics_out,
        test_preds_out=test_preds_out,
        errors_out=errors_out,
        feature_importance_out=feature_importance_out,
        metadata_out=metadata_out,
        dataset_info=dataset_info,
        tuned_params=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train appears_in_region model with CV, tuning, and multiple models"
    )
    parser.add_argument(
        "--input",
        default="src/data/train_table.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--metrics-out",
        default="src/data/model_metrics.json",
        help="Where to write JSON metrics",
    )
    parser.add_argument(
        "--model-out",
        default="src/data/model.joblib",
        help="Where to write selected model artifact",
    )
    parser.add_argument(
        "--region-metrics-out",
        default="src/data/region_metrics.csv",
        help="Where to write per-region evaluation CSV",
    )
    parser.add_argument(
        "--test-preds-out",
        default="src/data/test_predictions.csv",
        help="Where to write test predictions with probabilities",
    )
    parser.add_argument(
        "--errors-out",
        default="src/data/error_rows.csv",
        help="Where to write false positive/false negative rows",
    )
    parser.add_argument(
        "--feature-importance-out",
        default="src/data/feature_importance.csv",
        help="Where to write feature importance CSV",
    )
    parser.add_argument(
        "--metadata-out",
        default="src/data/model_metadata.json",
        help="Where to write model metadata JSON (for frontend)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test split",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction for validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    train_and_evaluate(
        input_csv=args.input,
        metrics_out=args.metrics_out,
        model_out=args.model_out,
        region_metrics_out=args.region_metrics_out,
        test_preds_out=args.test_preds_out,
        errors_out=args.errors_out,
        test_size=args.test_size,
        seed=args.seed,
        val_size=args.val_size,
        feature_importance_out=args.feature_importance_out,
        metadata_out=args.metadata_out,
    )


if __name__ == "__main__":
    main()
