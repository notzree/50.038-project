from pathlib import Path
import json
import requests

import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"

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

DATA_COLUMNS = ["title", "artist", "date", "region", "rank", "streams", "trend"]

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
    path = DATA_DIR / "mvp_region_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_chart_sample(song_query: str, region_query: str, period: str, max_rows: int = 2000) -> pd.DataFrame:
    path = DATA_DIR / "charts.csv"
    if not path.exists():
        return pd.DataFrame(columns=DATA_COLUMNS)

    song_query = song_query.strip().lower()
    region_query = region_query.strip().lower()

    start_date = None
    end_date = None
    if period == "2017-2018":
        start_date = pd.Timestamp("2017-01-01")
        end_date = pd.Timestamp("2018-12-31")
    elif period == "2019-2020":
        start_date = pd.Timestamp("2019-01-01")
        end_date = pd.Timestamp("2020-12-31")
    elif period == "2021":
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

    frames = []
    total = 0
    for chunk in pd.read_csv(path, usecols=DATA_COLUMNS, parse_dates=["date"], chunksize=200000):
        if song_query:
            mask = (
                chunk["title"].str.lower().str.contains(song_query, na=False)
                | chunk["artist"].str.lower().str.contains(song_query, na=False)
            )
            chunk = chunk.loc[mask]
        if region_query:
            chunk = chunk.loc[chunk["region"].str.lower().str.contains(region_query, na=False)]
        if start_date is not None:
            chunk = chunk.loc[(chunk["date"] >= start_date) & (chunk["date"] <= end_date)]

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
    return results.sort_values(["date", "region", "rank"])


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
    max_score = map_df["score"].max()
    if max_score <= 0:
        map_df["opacity"] = 40
        map_df["normalized_score"] = 0.0
        return map_df

    map_df["normalized_score"] = map_df["score"] / max_score
    map_df["opacity"] = ((0.2 + map_df["normalized_score"] * 0.75) * 255).astype(int).clip(40, 255)
    return map_df


def build_map_df(region_metrics: pd.DataFrame, chart_df: pd.DataFrame) -> pd.DataFrame:
    if not chart_df.empty:
        popularity = (
            chart_df.groupby("region")["streams"]
            .sum()
            .rename("total_streams")
            .to_frame()
            .reset_index()
        )
        popularity["point_count"] = chart_df.groupby("region").size().values
        popularity["score"] = popularity["total_streams"]
        map_df = popularity
    elif not region_metrics.empty:
        map_df = region_metrics.rename(columns={"support": "point_count", "positive_rate": "score"})
        map_df["total_streams"] = 0
    else:
        return pd.DataFrame(columns=["region", "iso", "total_streams", "point_count", "score", "normalized_score", "opacity"])

    map_df["iso"] = map_df["region"].map(REGION_TO_ISO)
    map_df = map_df.dropna(subset=["iso"]).copy()
    return normalize_map_scores(map_df)


def build_geojson_overlay(map_df: pd.DataFrame, country_geojson: dict) -> dict:
    if not country_geojson or map_df.empty:
        return country_geojson

    iso_to_row = map_df.set_index("iso").to_dict(orient="index")
    features = []
    for feature in country_geojson.get("features", []):
        # Natural Earth uses ISO_A3 for ISO codes
        iso = feature["properties"].get("ISO_A3") or feature["properties"].get("ISO3166-1-Alpha-3")
        props = feature["properties"].copy()
        if iso in iso_to_row:
            row = iso_to_row[iso]
            props["score"] = float(row["score"])
            props["total_streams"] = int(row.get("total_streams", 0))
            props["point_count"] = int(row.get("point_count", 0))
            props["fill_color"] = [70, 130, 255, int(row["opacity"])]
        else:
            props["score"] = 0.0
            props["total_streams"] = 0
            props["point_count"] = 0
            props["fill_color"] = [200, 200, 200, 30]
        features.append({"type": "Feature", "geometry": feature["geometry"], "properties": props})
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
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        opacity=1.0,
        auto_highlight=True,
    )


def build_deck(geojson_overlay: dict, map_df: pd.DataFrame) -> pdk.Deck:
    if geojson_overlay.get("features") is None:
        return pdk.Deck(initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1.0, pitch=0), layers=[])

    if not map_df.empty:
        center = map_df.loc[map_df["score"].idxmax()]
        latitude = float(center.get("lat", 20.0)) if "lat" in center else 20.0
        longitude = float(center.get("lon", 0.0)) if "lon" in center else 0.0
        zoom = 1.2
    else:
        latitude, longitude, zoom = 20.0, 0.0, 1.0

    return pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0),
        layers=[build_map_layer(geojson_overlay)],
        tooltip={
            "text": "{properties.NAME}\nScore: {properties.score}\nStreams: {properties.total_streams}\nPoints: {properties.point_count}",
        },
    )


def render_legend() -> None:
    legend_html = """
    <div style='border:1px solid #ddd; padding:12px; border-radius:8px; max-width:280px;'>
      <div style='font-weight:600; margin-bottom:8px;'>Map legend</div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.95); border-radius:4px;'></div>
        <div>High popularity intensity</div>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.65); border-radius:4px;'></div>
        <div>Medium popularity intensity</div>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
        <div style='width:24px; height:16px; background:rgba(70,130,255,0.25); border-radius:4px;'></div>
        <div>Lower popularity intensity</div>
      </div>
      <div style='color:#555; font-size:12px; margin-top:10px;'>Opacity encodes region popularity strength. Brighter fill means stronger signal in the current filter.</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


st.set_page_config(
    page_title="50.038 Project - Song Virality Trends",
    page_icon="🎵",
    layout="wide",
)

st.title("Song Virality Trends over Region and Time")
st.subheader("By Richard, Daniel, Maddie, and Maggie")

charts_available = (DATA_DIR / "charts.csv").exists()
if not charts_available:
    st.warning("Charts dataset not found in src/data/charts.csv. Run the pipeline first to generate chart data.")

region_metrics = load_region_metrics()
regions = sorted(region_metrics["region"].unique()) if not region_metrics.empty else []

st.sidebar.header("Search filters")
st.sidebar.write("Filter the world map and popularity summary by song, artist, region, or timeframe.")

song_name = st.sidebar.text_input("Song or artist", placeholder="Shape of You")
region_query = st.sidebar.text_input("Region name", placeholder="Australia, Brazil, United Kingdom")
time_period = st.sidebar.selectbox(
    "Time period",
    ["All data", "2017-2018", "2019-2020", "2021"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Available region preview")
if regions:
    st.sidebar.write(", ".join(regions[:8]) + (", ..." if len(regions) > 8 else ""))
else:
    st.sidebar.write("No region metrics available yet.")

filtered_df = load_chart_sample(song_name, region_query, time_period)
map_df = build_map_df(region_metrics, filtered_df)
country_geojson = load_country_geojson()
geojson_overlay = build_geojson_overlay(map_df, country_geojson)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🗺️ World Map", "📈 Time Series", "📊 Popularity Index"])

with tab1:
    st.subheader("World popularity trends by region")
    if not geojson_overlay or not geojson_overlay.get("features"):
        st.info("No map data is available. Check that src/app/countries.geojson exists and that chart or region metrics data is present.")
    else:
        st.pydeck_chart(build_deck(geojson_overlay, map_df))
        with st.expander("Map legend"):
            render_legend()
        st.markdown("### Region details")
        st.dataframe(
            map_df[["region", "point_count", "total_streams", "score", "normalized_score"]]
            .sort_values("score", ascending=False)
            .head(30),
            width="stretch",
        )

with tab2:
    st.subheader("Streams over time")
    time_df = build_time_series(filtered_df)
    if time_df.empty:
        st.info("No time-series data available for the current filter.")
    else:
        st.line_chart(time_df)

with tab3:
    st.subheader("Popularity by region")
    popularity_df = build_popularity_index(filtered_df)
    if popularity_df.empty:
        st.info("No popularity summary available for the current filter.")
    else:
        st.dataframe(popularity_df.head(50), width="stretch")
        st.markdown("### Top tracks in the filtered subset")
        top_tracks = (
            filtered_df.groupby(["title", "artist"])["streams"]
            .sum()
            .reset_index()
            .sort_values("streams", ascending=False)
            .head(20)
        )
        st.dataframe(top_tracks, width="stretch")
