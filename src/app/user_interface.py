from pathlib import Path
import json
import sys
import tempfile
import requests

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_features import extract_basic_features, extract_full_features
from high_level_features import compute_high_level_features

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
PLOTS_DIR = DATA_DIR / "plots"

REGION_TO_ISO = {
    "Andorra": "AND",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Chile": "CHL",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hong Kong": "HKG",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "Indonesia": "IDN",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Malaysia": "MYS",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Singapore": "SGP",
    "Slovakia": "SVK",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Taiwan": "TWN",
    "Thailand": "THA",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Vietnam": "VNM",
}

DATA_COLUMNS = [
    "track_id",
    "title",
    "artist",
    "date",
    "region",
    "rank",
    "streams",
    "trend",
]
CHART_READ_COLUMNS = [
    "url",
    "title",
    "artist",
    "date",
    "region",
    "rank",
    "streams",
    "trend",
]

PLOT_TITLES = {
    "01_dataset_overview.png": "Dataset Overview",
    "02_label_distribution.png": "Label Distribution",
    "03_feature_distributions.png": "Feature Distributions",
    "04_model_comparison.png": "Model Comparison",
    "05_confusion_matrices.png": "Confusion Matrices",
    "06_region_performance.png": "Region Performance",
    "07_region_errors.png": "Region Error Counts",
    "08_region_feature_lift_heatmap.png": "Region Feature Lift Heatmap",
    "09_region_probability_gap.png": "Region Probability Gap",
    "10_trends_coverage_heatmap.png": "Trends Coverage Heatmap",
    "11_trends_distributions_by_label.png": "Trends by Label",
    "12_trends_ablation_comparison.png": "Trends Ablation Comparison",
    "13_trends_error_reduction_by_region.png": "Trends Error Reduction",
    "14_pr_curve.png": "Precision-Recall Curve",
    "15_calibration_curve.png": "Calibration Curve",
    "16_learning_curve.png": "Learning Curve",
}


@st.cache_data
def load_country_geojson() -> dict:
    try:
        response = requests.get(GEOJSON_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Failed to load country GeoJSON: {e}")
        return {}


@st.cache_data
def load_region_metrics() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "region_metrics.csv",
        DATA_DIR / "mvp_region_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_chart_sample(
    track_id_query: str,
    region_query: str,
    max_rows: int = 2000,
) -> pd.DataFrame:
    path = DATA_DIR / "charts.csv"
    if not path.exists():
        return pd.DataFrame(columns=DATA_COLUMNS)

    track_id_query = track_id_query.strip().lower()
    region_query = region_query.strip().lower()

    frames = []
    total = 0
    for chunk in pd.read_csv(
        path, usecols=CHART_READ_COLUMNS, parse_dates=["date"], chunksize=200000
    ):
        chunk["track_id"] = (
            chunk["url"].astype(str).str.split("?").str[0].str.rsplit("/", n=1).str[-1]
        )
        if track_id_query:
            chunk = chunk.loc[chunk["track_id"].str.lower() == track_id_query]
        if region_query:
            chunk = chunk.loc[
                chunk["region"].str.lower().str.contains(region_query, na=False)
            ]

        if not chunk.empty:
            frames.append(chunk)
            total += len(chunk)
            if total >= max_rows:
                break

    if not frames:
        return pd.DataFrame(columns=DATA_COLUMNS)

    results = pd.concat(frames, ignore_index=True)
    if len(results) > max_rows:
        results = results.head(max_rows)
    return results[DATA_COLUMNS].sort_values(["date", "region", "rank"])


@st.cache_data
def load_high_level_features() -> pd.DataFrame:
    path = DATA_DIR / "audio_features_high_level.csv"
    if not path.exists():
        return pd.DataFrame(columns=["track_id"])
    df = pd.read_csv(path)
    if "track_id" not in df.columns:
        return pd.DataFrame(columns=["track_id"])
    return df.drop_duplicates(subset=["track_id"], keep="first")


@st.cache_data
def load_train_table_for_map() -> pd.DataFrame:
    path = DATA_DIR / "train_table.csv"
    if not path.exists():
        return pd.DataFrame(columns=["track_id", "region", "appears_in_region"])
    df = pd.read_csv(path)
    required = {"track_id", "region", "appears_in_region"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["track_id", "region", "appears_in_region"])
    return df


def load_plot_files() -> list[Path]:
    if not PLOTS_DIR.exists():
        return []
    return sorted(PLOTS_DIR.glob("*.png"), key=lambda p: p.name)


@st.cache_resource
def load_model_and_metadata(model_path: str) -> tuple[object, dict | None]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    clf = joblib.load(model_file)
    metadata_path = model_file.with_name("model_metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    return clf, metadata


@st.cache_data
def load_country_profile() -> pd.DataFrame | None:
    path = DATA_DIR / "country_profile_features.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "region" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _country_features_for_region(
    country_df: pd.DataFrame | None,
    region: str,
    expected_cols: set[str],
) -> dict:
    if country_df is None:
        return {}

    match = country_df[
        country_df["region"].astype(str).str.strip().str.lower()
        == region.strip().lower()
    ]
    if match.empty:
        return {}

    row = match.iloc[0].to_dict()
    exclude = {"region", "iso3", "data_year", "source_notes"}
    out = {}
    for key, value in row.items():
        if key in exclude or key not in expected_cols:
            continue
        out[key] = value

    if "country_profile_missing" in expected_cols:
        out["country_profile_missing"] = 0
    return out


def predict_uploaded_track_map(
    uploaded_audio,
    regions: list[str],
    model_path: str,
    threshold_override: float | None = None,
) -> tuple[pd.DataFrame, float, int, dict]:
    clf, metadata = load_model_and_metadata(model_path)

    suffix = Path(uploaded_audio.name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.getvalue())
        temp_audio_path = tmp.name

    try:
        if metadata and metadata.get("feature_set") == "full":
            base_features = extract_full_features(temp_audio_path)
        else:
            base_features = extract_basic_features(temp_audio_path)
    finally:
        Path(temp_audio_path).unlink(missing_ok=True)

    high_level_features = compute_high_level_features(base_features)
    expected_cols = set(metadata.get("feature_columns", [])) if metadata else set()
    model_default_threshold = (
        float(metadata.get("best_threshold", 0.5)) if metadata else 0.5
    )
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else model_default_threshold
    )
    country_df = load_country_profile()

    rows = []
    for region in regions:
        row = {"region": region, **base_features, **high_level_features}

        if metadata and "primary_genre" in expected_cols:
            row["primary_genre"] = "unknown"

        row.update(_country_features_for_region(country_df, region, expected_cols))

        if metadata:
            for col in expected_cols:
                if col not in row:
                    if col == "country_profile_missing":
                        row[col] = 1
                    else:
                        row[col] = np.nan
        rows.append(row)

    X = pd.DataFrame(rows)
    if metadata and metadata.get("feature_columns"):
        X = X[metadata["feature_columns"]]

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    pred_df = pd.DataFrame(
        {
            "region": regions,
            "score": probs,
            "prediction": preds,
            "point_count": 1,
        }
    )
    return pred_df, threshold, int(preds.sum()), high_level_features


def get_high_level_feature_columns(high_level_df: pd.DataFrame) -> list[str]:
    if high_level_df.empty:
        return []
    return sorted(
        [c for c in high_level_df.columns if c != "track_id" and c.endswith("_proxy")]
    )


def filter_chart_by_high_level_feature(
    chart_df: pd.DataFrame,
    high_level_df: pd.DataFrame,
    feature_name: str,
    feature_range: tuple[float, float],
) -> pd.DataFrame:
    if chart_df.empty or high_level_df.empty or not feature_name:
        return chart_df
    if feature_name not in high_level_df.columns:
        return chart_df

    lo, hi = feature_range
    feature_df = high_level_df[["track_id", feature_name]].dropna(subset=[feature_name])
    merged = chart_df.merge(feature_df, on="track_id", how="inner")
    return merged[(merged[feature_name] >= lo) & (merged[feature_name] <= hi)].copy()


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("date")["streams"]
        .sum()
        .rename("total_streams")
        .to_frame()
        .reset_index()
        .set_index("date")
    )


def build_popularity_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("region")
        .agg(
            songs_in_region=("title", "nunique"),
            avg_rank=("rank", "mean"),
            avg_streams=("streams", "mean"),
            total_streams=("streams", "sum"),
        )
        .reset_index()
        .sort_values("total_streams", ascending=False)
    )
    summary["avg_rank"] = summary["avg_rank"].round(1)
    summary["avg_streams"] = summary["avg_streams"].round(0).astype(int)
    summary["total_streams"] = summary["total_streams"].astype(int)
    return summary


def normalize_map_scores(map_df: pd.DataFrame) -> pd.DataFrame:
    if map_df.empty:
        return map_df

    scores = pd.to_numeric(map_df["score"], errors="coerce").fillna(0.0)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max <= score_min:
        map_df["normalized_score"] = 0.0
        map_df["opacity"] = 70
        return map_df

    minmax = (scores - score_min) / (score_max - score_min)
    rank_pct = scores.rank(method="average", pct=True)
    blended = (0.45 * minmax) + (0.55 * rank_pct)

    # Nonlinear contrast makes nearby probabilities (e.g. 0.15 vs 0.21) more visible.
    visual_score = blended.pow(0.35).clip(0.0, 1.0)
    map_df["normalized_score"] = visual_score
    map_df["opacity"] = ((0.3 + visual_score * 0.7) * 255).astype(int).clip(70, 255)
    return map_df


def build_map_df(
    region_metrics: pd.DataFrame,
    threshold: float,
    upload_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if region_metrics.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "iso",
                "point_count",
                "score",
                "normalized_score",
                "opacity",
            ]
        )

    map_df = region_metrics.rename(
        columns={"support": "point_count", "positive_rate": "score"}
    ).copy()

    if upload_predictions is not None and not upload_predictions.empty:
        pred_cols = ["region", "score", "point_count"]
        map_df = map_df.drop(columns=["point_count", "score"], errors="ignore")
        map_df = map_df.merge(upload_predictions[pred_cols], on="region", how="left")
        map_df["point_count"] = map_df["point_count"].fillna(0).astype(int)
        map_df["score"] = map_df["score"].fillna(0.0)
        map_df["iso"] = map_df["region"].map(REGION_TO_ISO)
        map_df = map_df.dropna(subset=["iso"]).copy()
        map_df = normalize_map_scores(map_df)
        map_df["above_threshold"] = map_df["score"] >= float(threshold)
        return map_df

    map_df["iso"] = map_df["region"].map(REGION_TO_ISO)
    map_df = map_df.dropna(subset=["iso"]).copy()
    map_df = normalize_map_scores(map_df)
    map_df["above_threshold"] = map_df["score"] >= float(threshold)
    return map_df


def build_geojson_overlay(map_df: pd.DataFrame, country_geojson: dict) -> dict:
    if not country_geojson or map_df.empty:
        return country_geojson

    iso_to_row = map_df.set_index("iso").to_dict(orient="index")
    features = []
    for feature in country_geojson.get("features", []):
        # Natural Earth uses ISO_A3 for ISO codes
        iso = feature["properties"].get("ISO_A3") or feature["properties"].get(
            "ISO3166-1-Alpha-3"
        )
        props = feature["properties"].copy()
        props["country_name"] = (
            props.get("NAME") or props.get("ADMIN") or props.get("name") or "Unknown"
        )
        if iso in iso_to_row:
            row = iso_to_row[iso]
            props["score"] = float(row["score"])
            props["point_count"] = int(row.get("point_count", 0))
            props["virality_level"] = float(row["score"])
            props["samples"] = int(row.get("point_count", 0))
            above_threshold = bool(row.get("above_threshold", False))
            props["above_threshold"] = above_threshold
            props["threshold_flag"] = "YES" if above_threshold else "NO"
            normalized = float(row.get("normalized_score", 0.0))
            red = int(236 - 225 * normalized)
            green = int(244 - 183 * normalized)
            blue = int(255 - 110 * normalized)
            props["fill_color"] = [red, green, blue, int(row["opacity"])]
            props["line_color"] = (
                [255, 120, 0, 230] if above_threshold else [255, 255, 255, 120]
            )
            props["line_width"] = 3 if above_threshold else 1
        else:
            props["score"] = 0.0
            props["point_count"] = 0
            props["virality_level"] = 0.0
            props["samples"] = 0
            props["above_threshold"] = False
            props["threshold_flag"] = "NO"
            props["fill_color"] = [200, 200, 200, 30]
            props["line_color"] = [255, 255, 255, 80]
            props["line_width"] = 1
        features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": props,
                "country_name": props["country_name"],
                "virality_level": props["virality_level"],
                "samples": props["samples"],
                "threshold_flag": props["threshold_flag"],
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_map_layer(geojson_overlay: dict) -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_overlay,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width="properties.line_width",
        line_width_min_pixels=1,
        opacity=1.0,
        auto_highlight=True,
    )


def build_deck(geojson_overlay: dict, map_df: pd.DataFrame) -> pdk.Deck:
    if geojson_overlay.get("features") is None:
        return pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=0, longitude=0, zoom=1.0, pitch=0
            ),
            layers=[],
        )

    if not map_df.empty:
        center = map_df.loc[map_df["score"].idxmax()]
        latitude = float(center.get("lat", 20.0)) if "lat" in center else 20.0
        longitude = float(center.get("lon", 0.0)) if "lon" in center else 0.0
        zoom = 1.2
    else:
        latitude, longitude, zoom = 20.0, 0.0, 1.0

    return pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=latitude, longitude=longitude, zoom=zoom, pitch=0
        ),
        layers=[build_map_layer(geojson_overlay)],
        tooltip={
            "text": "{country_name}\nVirality level: {virality_level}\nAbove threshold: {threshold_flag}\nSamples: {samples}",
        },
    )


def render_legend() -> None:
    legend_html = """
    <div style='border:1px solid #ddd; padding:12px; border-radius:8px; max-width:280px;'>
      <div style='font-weight:600; margin-bottom:8px;'>Map legend</div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.95); border-radius:4px;'></div>
        <div>High virality level</div>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.65); border-radius:4px;'></div>
        <div>Medium virality level</div>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.25); border-radius:4px;'></div>
        <div>Lower virality level</div>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.35); border:2px solid rgb(255,120,0); border-radius:4px;'></div>
        <div>Orange border = above threshold</div>
      </div>
      <div style='color:#555; font-size:12px; margin-top:10px;'>Opacity encodes regional virality level (positive rate). Brighter fill means stronger virality signal.</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


st.set_page_config(
    page_title="50.038 Project - Song Virality Trends",
    page_icon="🎵",
    layout="wide",
)

st.title("Song Virality Across Regions")
st.subheader("By Richard, Daniel, Maddie, and Maggie")

charts_available = (DATA_DIR / "charts.csv").exists()
if not charts_available:
    st.warning(
        "Charts dataset not found in src/data/charts.csv. Run the pipeline first to generate chart data."
    )

region_metrics = load_region_metrics()
regions = sorted(region_metrics["region"].unique()) if not region_metrics.empty else []

st.sidebar.header("Search filters")
st.sidebar.write("Upload audio to run model prediction by region.")

audio_file = st.sidebar.file_uploader(
    "Audio file (.mp3)", type=["mp3"], help="Upload a file named like <track_id>.mp3"
)
if audio_file:
    st.sidebar.audio(audio_file, format="audio/mpeg")
    st.sidebar.caption(f"Uploaded file: {audio_file.name}")

train_df = load_train_table_for_map()

model_default_threshold = 0.35
metadata_path = DATA_DIR / "model_metadata.json"
if metadata_path.exists():
    try:
        model_default_threshold = float(
            json.loads(metadata_path.read_text()).get("best_threshold", 0.35)
        )
    except Exception:
        pass

prediction_threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(model_default_threshold),
    step=0.01,
    help="Regions with predicted probability >= threshold are counted as positive.",
)
if not regions and not train_df.empty:
    regions = sorted(train_df["region"].dropna().astype(str).unique())

upload_predictions = None
upload_threshold = None
upload_positive_regions = None
uploaded_high_level_features = None
if audio_file and regions:
    try:
        (
            upload_predictions,
            upload_threshold,
            upload_positive_regions,
            uploaded_high_level_features,
        ) = predict_uploaded_track_map(
            uploaded_audio=audio_file,
            regions=regions,
            model_path=str(DATA_DIR / "model.joblib"),
            threshold_override=prediction_threshold,
        )
    except Exception as exc:
        st.error(f"Could not run model prediction for uploaded audio: {exc}")

map_df = build_map_df(
    region_metrics=region_metrics,
    threshold=prediction_threshold,
    upload_predictions=upload_predictions,
)
country_geojson = load_country_geojson()
geojson_overlay = build_geojson_overlay(map_df, country_geojson)

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ World Map", "📉 Generated Visualizations"])

with tab1:
    st.subheader("World virality level by region")
    if upload_predictions is not None:
        min_prob = float(upload_predictions["score"].min())
        max_prob = float(upload_predictions["score"].max())
        st.success(
            f"Uploaded-audio prediction active across {len(upload_predictions)} regions "
            f"(threshold={upload_threshold:.2f}, regions above threshold={upload_positive_regions}, "
            f"min_prob={min_prob:.4f}, max_prob={max_prob:.4f})."
        )
        if max_prob == min_prob:
            st.info(
                "All regions received the same probability for this upload, so the map appears uniform."
            )
    if not geojson_overlay or not geojson_overlay.get("features"):
        st.info(
            "No map data is available. Check that src/app/countries.geojson exists and that chart or region metrics data is present."
        )
    else:
        map_version = f"base-{len(map_df)}"
        if upload_predictions is not None and not upload_predictions.empty:
            map_version = (
                f"upload-{len(upload_predictions)}-"
                f"{float(upload_predictions['score'].max()):.6f}-"
                f"{float(upload_predictions['score'].min()):.6f}-"
                f"{upload_threshold:.2f}"
            )
        st.pydeck_chart(
            build_deck(geojson_overlay, map_df), key=f"world-map-{map_version}"
        )
        with st.expander("Map legend"):
            render_legend()
        st.markdown("### Region details")
        region_details = map_df.copy()
        region_details["virality_level"] = region_details["score"].astype(float)
        region_details["delta_vs_threshold"] = region_details["virality_level"] - float(
            prediction_threshold
        )
        region_details["above_threshold"] = np.where(
            region_details["virality_level"] >= float(prediction_threshold),
            "YES",
            "NO",
        )
        display_cols = [
            "region",
            "virality_level",
            "delta_vs_threshold",
            "above_threshold",
            "normalized_score",
            "point_count",
        ]
        for col in ["f1", "precision", "recall", "accuracy"]:
            if col in region_details.columns:
                display_cols.append(col)
        st.dataframe(
            region_details[display_cols].sort_values("virality_level", ascending=False),
            width="stretch",
        )
        if upload_predictions is not None:
            st.markdown("### Uploaded audio: top predicted regions")
            upload_table = upload_predictions.copy()
            upload_table["delta_vs_threshold"] = upload_table["score"] - float(
                prediction_threshold
            )
            upload_table["above_threshold"] = np.where(
                upload_table["score"] >= float(prediction_threshold), "YES", "NO"
            )
            if not region_metrics.empty:
                metrics_cols = [
                    c
                    for c in ["region", "positive_rate", "f1", "precision", "recall"]
                    if c in region_metrics.columns
                ]
                if metrics_cols:
                    upload_table = upload_table.merge(
                        region_metrics[metrics_cols], on="region", how="left"
                    ).rename(columns={"positive_rate": "baseline_positive_rate"})
            st.dataframe(
                upload_table.sort_values("score", ascending=False)
                .head(20)
                .rename(columns={"score": "predicted_probability"}),
                width="stretch",
            )
            if uploaded_high_level_features:
                st.markdown("### Uploaded audio: high-level features")
                hl_df = (
                    pd.DataFrame(
                        [
                            {"feature": k, "value": v}
                            for k, v in uploaded_high_level_features.items()
                        ]
                    )
                    .sort_values("feature")
                    .reset_index(drop=True)
                )
                numeric_vals = pd.to_numeric(hl_df["value"], errors="coerce")
                hl_df["value"] = np.where(
                    numeric_vals.notna(),
                    numeric_vals.round(4).astype(str),
                    hl_df["value"].astype(str),
                )
                st.dataframe(hl_df, width="stretch")

with tab2:
    st.subheader("Generated model visualizations")
    plot_files = load_plot_files()
    if not plot_files:
        st.info(
            "No generated plots found in src/data/plots. Run `uv run python src/make_visualizations.py` first."
        )
    else:
        selected_name = st.selectbox(
            "Visualization",
            options=[p.name for p in plot_files],
            index=0,
            format_func=lambda name: PLOT_TITLES.get(name, name),
        )
        selected_path = PLOTS_DIR / selected_name
        st.image(
            str(selected_path),
            caption=PLOT_TITLES.get(selected_name, selected_name),
            use_container_width=True,
        )

        with st.expander("Show all available plots"):
            for p in plot_files:
                st.write(f"- {PLOT_TITLES.get(p.name, p.name)} (`{p.name}`)")
