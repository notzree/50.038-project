import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    categorical_features = ["region"] if "region" in feature_cols else []
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


def _evaluate_model(clf: Pipeline, X_test: pd.DataFrame, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
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


def _build_region_metrics(clf: Pipeline, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    results = X_test.copy()
    results["y_true"] = y_test
    results["y_pred"] = clf.predict(X_test)

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


def train_and_evaluate(
    input_csv: str,
    metrics_out: str,
    model_out: str,
    region_metrics_out: str,
    test_preds_out: str,
    errors_out: str,
    test_size: float,
    seed: int,
) -> dict:
    print(f"Loading training table: {input_csv}")
    df = pl.read_csv(input_csv)

    required_cols = {"track_id", "region", "appears_in_region"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = [c for c in df.columns if c not in {"track_id", "appears_in_region"}]
    if not feature_cols:
        raise ValueError("No feature columns found in input table")

    all_cols = df.columns
    data_pd = pd.DataFrame({col: df[col].to_list() for col in all_cols})
    X = data_pd[feature_cols]
    y = data_pd["appears_in_region"].to_numpy()
    meta = data_pd[["track_id", "region"]]

    print(
        f"Rows={df.height}, features={len(feature_cols)}, positive_rate={float(y.mean()):.4f}"
    )

    preprocessor = _build_preprocessor(feature_cols)

    logistic_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print("Training logistic regression...")
    logistic_clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", logistic_model)]
    )
    logistic_clf.fit(X_train, y_train)
    logistic_metrics = _evaluate_model(logistic_clf, X_test, y_test)

    print("Training random forest...")
    rf_clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf_model)])
    rf_clf.fit(X_train, y_train)
    rf_metrics = _evaluate_model(rf_clf, X_test, y_test)

    model_candidates = {
        "logistic_regression": {"pipeline": logistic_clf, "metrics": logistic_metrics},
        "random_forest": {"pipeline": rf_clf, "metrics": rf_metrics},
    }

    best_model_name = max(
        model_candidates,
        key=lambda name: model_candidates[name]["metrics"]["f1"],
    )
    best_clf = model_candidates[best_model_name]["pipeline"]
    print(f"Selected best model by F1: {best_model_name}")

    best_y_prob = best_clf.predict_proba(X_test)[:, 1]
    default_threshold_metrics = _evaluate_at_threshold(y_test, best_y_prob, 0.5)
    tuned_threshold_metrics = _find_best_threshold(y_test, best_y_prob)

    pred_df = meta_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_prob"] = best_y_prob
    pred_df["y_pred_default_0_5"] = (best_y_prob >= 0.5).astype(int)
    tuned_threshold = tuned_threshold_metrics["threshold"]
    pred_df["y_pred_tuned"] = (best_y_prob >= tuned_threshold).astype(int)

    pred_path = Path(test_preds_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

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
    error_by_region_path = errors_path.with_name("mvp_error_counts_by_region.csv")
    error_by_region.to_csv(error_by_region_path, index=False)

    region_metrics_df = _build_region_metrics(best_clf, X_test, y_test)
    region_metrics_path = Path(region_metrics_out)
    region_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    region_metrics_df.to_csv(region_metrics_path, index=False)

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, model_path)

    metrics = {
        "dataset": {
            "num_rows": int(df.height),
            "num_features": len(feature_cols),
            "test_size": test_size,
            "seed": seed,
            "positive_rate": float(y.mean()),
        },
        "models": {
            "logistic_regression": logistic_metrics,
            "random_forest": rf_metrics,
        },
        "selected_model": best_model_name,
        "selected_model_thresholding": {
            "default_0_5": default_threshold_metrics,
            "best_f1_threshold_on_test": tuned_threshold_metrics,
        },
        "artifacts": {
            "model_path": str(model_path),
            "region_metrics_path": str(region_metrics_path),
            "test_predictions_path": str(pred_path),
            "error_rows_path": str(errors_path),
            "error_counts_by_region_path": str(error_by_region_path),
        },
    }

    out_path = Path(metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote metrics to {metrics_out}")
    print(f"Wrote best model to {model_out}")
    print(f"Wrote region metrics to {region_metrics_out}")
    print(f"Wrote test predictions to {test_preds_out}")
    print(f"Wrote error rows to {errors_out}")
    print(f"Wrote error counts by region to {error_by_region_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MVP appears_in_region baseline model"
    )
    parser.add_argument(
        "--input",
        default="src/data/train_table_mvp.csv",
        help="Path to MVP training CSV",
    )
    parser.add_argument(
        "--metrics-out",
        default="src/data/mvp_metrics.json",
        help="Where to write JSON metrics",
    )
    parser.add_argument(
        "--model-out",
        default="src/data/mvp_model.joblib",
        help="Where to write selected model artifact",
    )
    parser.add_argument(
        "--region-metrics-out",
        default="src/data/mvp_region_metrics.csv",
        help="Where to write per-region evaluation CSV for selected model",
    )
    parser.add_argument(
        "--test-preds-out",
        default="src/data/mvp_test_predictions.csv",
        help="Where to write test predictions with probabilities",
    )
    parser.add_argument(
        "--errors-out",
        default="src/data/mvp_error_rows.csv",
        help="Where to write false positive/false negative rows",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test split",
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
    )


if __name__ == "__main__":
    main()
