"""
google_trens_collector.py - REMASTERED
--------------------------------------
Collects Google Trends search signals for songs in the dataset
- produces a feature table to join onto existing training dataset
train_table.csv before model training

script output: src/data/trends_features.csv

dataset columns:
- song_id: unique identifier for the song, should match audio_manifest
- region: tw-letter country code (e.g. US, KR) or 'global' for worldwide trends
- trends_week: ISO week string (e.g. 2023-W01) representing the week of the trend data
gt_peak:peak search interest across trailing twelve weeks
gt_mean: mean search interest across trailing twelve weeks
gt_slope: linear regression slope
gt_momentum: interest in final week minus interest in first week
gt_weeks_above50: counter of weeks where interest exceeded 50/100

Usage (standalone): uv run python src/google_trends_collector.py --manifest src/data/audio_manifest.csv --output src/data/trends_features.csv

Usage (called from pipeline script):
    from google_trends_collector import build_trends_features
    df = build_trends_features(charts_path, manifest_path, output_path)
"""

from __future__ import annotations

import time
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
from pytrends.request import TrendReq

###LOGGING###
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("trends_collector")

# Region mapping: charts.csv "region" name to Google Trends geo code
REGION_TO_GEO: dict[str, str] = {
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bolivia": "BO",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Canada": "CA",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Global": "",  # "" = worldwide in pytrends
    "Greece": "GR",
    "Guatemala": "GT",
    "Honduras": "HN",
    "Hong Kong": "HK",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malaysia": "MY",
    "Mexico": "MX",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Norway": "NO",
    "Panama": "PA",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Singapore": "SG",
    "Slovakia": "SK",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Thailand": "TH",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "GB",
    "United States": "US",
    "Uruguay": "UY",
    "Vietnam": "VN",
}

###CONSTANTS###
DATA_DIR = Path("src/data")
CHARTS_CSV = DATA_DIR / "charts.csv"
MANIFEST_CSV = DATA_DIR / "audio_manifest.csv"
OUTPUT_CSV = DATA_DIR / "trends_features.csv"

TRAILING_WEEKS = 12
SLEEP_SECONDS = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 30  # Exponential backoff factor for retries


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_query(track_name: str, artist_name: str) -> str:
    """
    Builds query string for Google Trends.
    Uses "Song Title Artist, but can swap to just track name
    """
    track = track_name[:40].strip()
    artist = artist_name[:30].strip()
    return sanitize_query(f"{track} {artist}")


def sanitize_query(text: str, max_len: int = 70) -> str:
    """Clean noisy query strings for pytrends stability.

    Keeps letters/digits/spaces and a small set of safe punctuation,
    collapses whitespace, and returns empty string for punctuation-only inputs.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # Remove most punctuation/symbol noise while keeping readable separators.
    s = re.sub(r"[^\w\s&'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # If query has no alphanumeric tokens after cleanup, skip it.
    if not re.search(r"[A-Za-z0-9]", s):
        return ""

    return s[:max_len].strip()


def _anchor_date_to_timeframe(
    anchor_date: datetime, n_weeks: int = TRAILING_WEEKS
) -> str:
    """
    Convert anchor date into pytrends timeframe string, trailing 12 weeks

    Returns e.g. "2024-01-01 2024-03-25"
    """
    end = anchor_date
    start = end - timedelta(weeks=n_weeks)
    return f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"


def _fetch_interest(
    pytrends: TrendReq,
    keyword: str,
    geo: str,
    timeframe: str,
) -> pd.Series | None:
    """Fetch weekly interest over time,
    Return pandas Series indexed by date or None or failure"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload(
                kw_list=[keyword], cat=35, timeframe=timeframe, geo=geo, gprop=""
            )
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                return None
            return df[keyword]
        except Exception as e:
            log.warning(
                "Attempt %d/%d failed for '%s' (%s): %s",
                attempt,
                MAX_RETRIES,
                keyword,
                geo,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))  # Exponential backoff
    return None


def _fetch_interest_bundle(
    pytrends: TrendReq,
    keyword: str,
    timeframe: str,
) -> tuple[pd.Series | None, pd.DataFrame | None]:
    """Fetch both interest-over-time and interest-by-region with one payload build."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload(
                kw_list=[keyword],
                cat=35,
                timeframe=timeframe,
                geo="",  # worldwide context, then split by region via interest_by_region
                gprop="",
            )
            iot = pytrends.interest_over_time()
            ibr = pytrends.interest_by_region(resolution="COUNTRY", inc_low_vol=True)

            series = None
            if iot is not None and not iot.empty and keyword in iot.columns:
                series = iot[keyword]

            region_df = None
            if ibr is not None and not ibr.empty:
                region_df = ibr

            return series, region_df
        except Exception as e:
            log.warning(
                "Attempt %d/%d failed for bundle '%s': %s",
                attempt,
                MAX_RETRIES,
                keyword,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))

    return None, None


def _region_interest_map(region_df: pd.DataFrame, keyword: str) -> dict[str, float]:
    """Map charts-style region names to Google Trends country interest values."""
    if region_df is None or region_df.empty:
        return {}

    # Determine value column robustly.
    value_col = keyword if keyword in region_df.columns else None
    if value_col is None:
        candidates = [c for c in region_df.columns if c not in {"isPartial", "geoCode"}]
        if not candidates:
            return {}
        value_col = candidates[0]

    # Reverse map geo code -> charts region name.
    code_to_region = {v: k for k, v in REGION_TO_GEO.items() if v}

    mapped: dict[str, float] = {}
    for idx, row in region_df.iterrows():
        score = _safe_float(row.get(value_col), 0.0)
        if score < 0:
            continue

        region_name = None

        # pytrends often uses country names as index.
        if isinstance(idx, str) and idx in REGION_TO_GEO:
            region_name = idx

        # fallback by geo code when available
        if region_name is None and "geoCode" in region_df.columns:
            code = str(row.get("geoCode", "")).strip().upper()
            region_name = code_to_region.get(code)

        if region_name is not None:
            mapped[region_name] = float(score)

    return mapped


def extract_features(series: pd.Series) -> dict[str, float]:
    # Derive summary features from weekly interest time-series
    vals = series.values.astype(float)
    if len(vals) == 0:
        return {
            "gt_peak": 0.0,
            "gt_mean": 0.0,
            "gt_slope": 0.0,
            "gt_momentum": 0.0,
            "gt_weeks_above50": 0,
        }

    x = np.arange(len(vals))
    slope = float(np.polyfit(x, vals, 1)[0]) if len(vals) > 1 else 0.0
    return {
        "gt_peak": float(np.max(vals)),
        "gt_mean": float(np.mean(vals)),
        "gt_slope": slope,
        "gt_momentum": float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0,
        "gt_weeks_above50": float((vals > 50).sum()),
    }


def build_dataset(
    song_id: str,
    search_term: str,
    geo: str = "",
    timeframe: str = "today 3-m",
) -> dict[str, pd.DataFrame]:
    """Small helper API for Streamlit: fetch raw trends tables for one query.

    Returns a dict with:
    - interest_over_time: dataframe with date index + query column
    - interest_by_region: dataframe with region scores
    """
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload([search_term], geo=geo, timeframe=timeframe)

    iot = pytrends.interest_over_time()
    if iot is None or iot.empty:
        iot = pd.DataFrame(columns=["date", search_term])
    else:
        iot = iot.reset_index()

    ibr = pytrends.interest_by_region(resolution="COUNTRY", inc_low_vol=True)
    if ibr is None or ibr.empty:
        ibr = pd.DataFrame(columns=["geoName", search_term])
    else:
        ibr = ibr.reset_index()

    iot["song_id"] = song_id
    ibr["song_id"] = song_id

    return {
        "interest_over_time": iot,
        "interest_by_region": ibr,
    }


"""
    build_trends_features
    For every (track_id, region) pair appearing in the training table,
    query Google Trends and produce a row of trend features
    
    Parameters:
    charts_path: path to chart CSV (track_name, artist, region, date)
    manifest_path: path to audio_manifest.csv (track_id, track_name, artist_name)
    output_path: path to write trends_features.csv
    trailing_weeks: weeks of histor per anchor date
    delay: seconds to sleep to not over-call google API
    
    Returns:
    plars.Dataframe with one row per track_id, region
    """


def build_trends_features(
    charts_path: Path = CHARTS_CSV,
    manifest_path: Path = MANIFEST_CSV,
    output_path: Path = OUTPUT_CSV,
    trailing_weeks: int = TRAILING_WEEKS,
    delay: float = SLEEP_SECONDS,
    max_pairs: int | None = None,
) -> pl.DataFrame:
    # load existing data
    log.info("Loading charts from %s", charts_path)
    charts = pl.read_csv(charts_path, try_parse_dates=True)

    log.info("Loading manifest from %s", manifest_path)
    manifest = pl.read_csv(manifest_path)

    charts = charts.rename({c: c.strip().lower() for c in charts.columns})
    manifest = manifest.rename({c: c.strip().lower() for c in manifest.columns})

    # Derive track_id from URL when needed (charts.csv typically has url, not track_id)
    if "track_id" not in charts.columns and "url" in charts.columns:
        charts = charts.with_columns(
            pl.col("url")
            .cast(pl.String)
            .str.split("?")
            .list.first()
            .str.split("/")
            .list.last()
            .alias("track_id")
        )

    if "track_id" in charts.columns and "track_id" in manifest.columns:
        join_key = "track_id"
    else:
        join_key = "track_name"
        log.warning("No 'track_id' in charts - joining on 'track_name'")

    chart_anchors = charts.group_by([join_key, "region"]).agg(
        pl.col("date").min().alias("anchor_date")
    )

    # Build metadata with robust fallbacks:
    # 1) from manifest if available, 2) from charts title/artist
    chart_meta = charts.group_by(join_key).agg(
        pl.col("title").first().alias("track_name"),
        pl.col("artist").first().alias("artist_name"),
    )

    manifest_meta_cols = [join_key]
    if "track_name" in manifest.columns:
        manifest_meta_cols.append("track_name")
    elif "track_title" in manifest.columns:
        manifest = manifest.rename({"track_title": "track_name"})
        manifest_meta_cols.append("track_name")

    if "artist_name" in manifest.columns:
        manifest_meta_cols.append("artist_name")
    elif "track_artist" in manifest.columns:
        manifest = manifest.rename({"track_artist": "artist_name"})
        manifest_meta_cols.append("artist_name")

    manifest_meta = manifest.select(manifest_meta_cols).unique(join_key)
    meta = chart_meta.join(manifest_meta, on=join_key, how="left", suffix="_manifest")
    if "track_name_manifest" in meta.columns:
        meta = meta.with_columns(
            pl.coalesce([pl.col("track_name_manifest"), pl.col("track_name")]).alias(
                "track_name"
            )
        ).drop("track_name_manifest")
    if "artist_name_manifest" in meta.columns:
        meta = meta.with_columns(
            pl.coalesce([pl.col("artist_name_manifest"), pl.col("artist_name")]).alias(
                "artist_name"
            )
        ).drop("artist_name_manifest")

    pairs = chart_anchors.join(meta, on=join_key, how="left")

    log.info("Collected %d (track, region) pairs to query.", len(pairs))

    # initialize pytrends and query loop
    pytrends = TrendReq(hl="en-US", tz=0)

    records: list[dict] = []

    # Convert pair table once (no pyarrow dependency)
    pairs_pd = pd.DataFrame(pairs.to_dicts())
    if pairs_pd.empty:
        result = pl.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output_path)
        return result

    # Query once per artist, then expand to all matching (track, region) rows.
    artist_meta = pairs_pd.groupby("artist_name", as_index=False).agg(
        track_name=("track_name", "first"),
        anchor_date=("anchor_date", "min"),
    )
    artist_meta["artist_name"] = artist_meta["artist_name"].fillna("").astype(str)
    artist_meta = artist_meta[artist_meta["artist_name"].str.len() > 0].copy()

    if max_pairs is not None and max_pairs > 0:
        # Keep compatibility with existing flag name; interpret as max artists to query.
        artist_meta = artist_meta.head(max_pairs)
        log.info("Limiting trends queries to first %d artists", len(artist_meta))

    pair_subset = pairs_pd[
        pairs_pd["artist_name"].fillna("").isin(artist_meta["artist_name"])
    ].copy()

    for i, row in artist_meta.iterrows():
        artist_name: str = str(row.get("artist_name", ""))
        fallback_track: str = str(row.get("track_name", ""))

        raw_date = row.get("anchor_date")
        try:
            anchor_dt = pd.to_datetime(raw_date)
        except Exception:
            anchor_dt = datetime.utcnow()

        keyword = sanitize_query(artist_name)
        if not keyword:
            keyword = sanitize_query(fallback_track)
        timeframe = _anchor_date_to_timeframe(anchor_dt, trailing_weeks)
        week_label = anchor_dt.strftime("%G-W%V")

        log.info(
            "[%d/%d] artist=%-45s | %s",
            i + 1,
            len(artist_meta),
            keyword[:45],
            timeframe,
        )

        if not keyword:
            log.warning(
                "Skipping empty/invalid trends query for artist '%s'", artist_name
            )
            series, by_region_df = None, None
        else:
            series, by_region_df = _fetch_interest_bundle(pytrends, keyword, timeframe)

        if series is not None:
            features = extract_features(series)
        else:
            log.warning(
                "No time-series data returned for %s; filling with zeros.", keyword
            )
            features = {
                "gt_peak": 0.0,
                "gt_mean": 0.0,
                "gt_slope": 0.0,
                "gt_momentum": 0.0,
                "gt_weeks_above50": 0.0,
            }

        region_scores = _region_interest_map(by_region_df, keyword)

        rows_for_artist = pair_subset[
            pair_subset["artist_name"].fillna("").astype(str) == artist_name
        ]
        for _, pair_row in rows_for_artist.iterrows():
            region_name = str(pair_row.get("region", "Global"))
            track_id = str(pair_row.get(join_key, ""))
            record = {
                join_key: track_id,
                "region": region_name,
                "trends_week": week_label,
                "gt_region_interest": float(region_scores.get(region_name, 0.0)),
                **features,
            }
            records.append(record)

        time.sleep(delay)

    result = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    log.info("Trends features written to %s", output_path)
    return result


def merge_trends_into_train_table(
    train_table_path: Path = DATA_DIR / "train_table.csv",
    trends_path: Path = OUTPUT_CSV,
    output_path: Path = DATA_DIR / "train_table_with_trends.csv",
) -> pl.DataFrame:
    """
    Left-join the trends features onto the existing train table.

    The resulting CSV is a drop-in replacement for train_table.csv;
    the new gt_* columns are simply extra features for the model.

    Rows with no trends match get 0-filled gt_* columns (safe default).
    """
    log.info("Loading train table from %s", train_table_path)
    train = pl.read_csv(train_table_path)
    train = train.rename({c: c.strip().lower() for c in train.columns})

    log.info("Loading trends features from %s", trends_path)
    trends = pl.read_csv(trends_path)
    trends = trends.rename({c: c.strip().lower() for c in trends.columns})

    # Determine join key(s)
    join_keys = ["region"]
    if "track_id" in train.columns and "track_id" in trends.columns:
        join_keys = ["track_id", "region"]
    elif "track_name" in train.columns and "track_name" in trends.columns:
        join_keys = ["track_name", "region"]

    # Drop the metadata-only column we don't need in train
    trends_feat_cols = join_keys + [c for c in trends.columns if c.startswith("gt_")]
    trends_slim = trends.select(trends_feat_cols)

    merged = train.join(trends_slim, on=join_keys, how="left")

    # Fill nulls in gt_* columns with 0.0
    gt_cols = [c for c in merged.columns if c.startswith("gt_")]
    merged = merged.with_columns([pl.col(c).fill_null(0.0) for c in gt_cols])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(output_path)
    log.info(
        "Merged train table saved (%d rows, %d cols) → %s",
        len(merged),
        len(merged.columns),
        output_path,
    )

    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Google Trends features for the virality model."
    )
    parser.add_argument("--charts", default=str(CHARTS_CSV), help="Path to charts.csv")
    parser.add_argument(
        "--manifest", default=str(MANIFEST_CSV), help="Path to audio_manifest.csv"
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_CSV), help="Where to write trends_features.csv"
    )
    parser.add_argument(
        "--weeks", type=int, default=TRAILING_WEEKS, help="Trailing weeks per query"
    )
    parser.add_argument(
        "--delay", type=float, default=SLEEP_SECONDS, help="Delay (s) between requests"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of artist-level queries",
    )
    parser.add_argument(
        "--merge", action="store_true", help="Also merge into train_table.csv"
    )
    args = parser.parse_args()

    build_trends_features(
        charts_path=Path(args.charts),
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        trailing_weeks=args.weeks,
        delay=args.delay,
        max_pairs=args.max_pairs,
    )

    if args.merge:
        merge_trends_into_train_table()
