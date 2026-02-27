import os
from multiprocessing import Semaphore

import kagglehub
import polars as pl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_dataset() -> str:
    """Download the dataset to a local path if it doesn't already exist. Returns the CSV path."""
    path = kagglehub.dataset_download(
        "dhruvildave/spotify-charts", path="charts.csv", output_dir=DATA_DIR
    )
    return path


def unify_title_url_mappings(csv_path: str) -> pl.LazyFrame:
    """Create a 1:1 mapping between (title, artist) and URL.

    Two problems exist in the raw data:
    1. Same URL, different title/artist strings (metadata drift)
    2. Same (title, artist), different URLs (re-releases, regional variants)

    We fix both:
    - Step 1: Canonicalize title/artist per URL (fixes metadata drift)
    - Step 2: Pick one canonical URL per (title, artist) (dedup re-releases)
    """
    lf = pl.scan_csv(csv_path)

    # Step 1: Normalize title/artist per URL
    url_canonical = lf.group_by("url").agg(
        pl.col("title").first().alias("canonical_title"),
        pl.col("artist").first().alias("canonical_artist"),
    )
    step1 = (
        lf.join(url_canonical, on="url", how="left")
        .drop("title", "artist")
        .rename({"canonical_title": "title", "canonical_artist": "artist"})
    )

    # Step 2: Pick one URL per (title, artist)
    title_artist_canonical = step1.group_by("title", "artist").agg(
        pl.col("url").first().alias("canonical_url")
    )
    step2 = (
        step1.join(title_artist_canonical, on=["title", "artist"], how="left")
        .drop("url")
        .rename({"canonical_url": "url"})
    )

    return step2


def get_mp3s_for_dataset(df: pl.DataFrame, max_concurrent: int) -> None:
    s = Semaphore(max_concurrent)
    spotify_ids = df.select("url")
