import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import kagglehub
import polars as pl
import yt_dlp

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SONGS_DIR = os.path.join(DATA_DIR, "songs")


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


CLIP_DURATION = 30  # seconds


def _download_one(title: str, artist: str, track_id: str) -> tuple[str, str | None]:
    """Download a 30s clip from the middle of a song via yt-dlp YouTube search.

    Returns (track_id, error_or_None).
    """
    final_path = os.path.join(SONGS_DIR, f"{track_id}.mp3")
    query = f"ytsearch1:{title} {artist}"

    # Step 1: Extract metadata to get duration
    extract_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(extract_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if info.get("entries"):
                info = info["entries"][0]

        duration = info.get("duration", 0)
        if duration <= CLIP_DURATION:
            # Song is shorter than 30s, just download the whole thing
            start_time = 0
        else:
            start_time = (duration - CLIP_DURATION) // 2

        # Step 2: Download with ffmpeg trimming from the middle
        download_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(SONGS_DIR, f"{track_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "download_ranges": yt_dlp.utils.download_range_func(
                None, [(start_time, start_time + CLIP_DURATION)]
            ),
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([info["webpage_url"]])

        return (track_id, None)
    except Exception as e:
        # Clean up partial files
        if os.path.exists(final_path):
            os.remove(final_path)
        return (track_id, str(e))


def get_mp3s_for_dataset(
    lf: pl.LazyFrame,
    max_workers: int = 12,
) -> None:
    """Download mp3s for all unique songs using yt-dlp YouTube search.

    Files are saved as {track-id}.mp3 in data/songs/.
    Already-downloaded tracks are skipped.
    Uses title + artist from the dataset to search YouTube directly (no Spotify API needed).
    """
    os.makedirs(SONGS_DIR, exist_ok=True)

    # Get unique (url, title, artist) — url is used to extract the track ID for filenames
    songs = lf.select("url", "title", "artist").unique(subset=["url"]).collect()

    # Filter out already-downloaded tracks
    existing = {
        f.removesuffix(".mp3") for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")
    }
    to_download = [
        row
        for row in songs.iter_rows(named=True)
        if row["url"].rsplit("/", 1)[-1] not in existing
    ]
    print(f"Downloading {len(to_download)} songs ({len(existing)} already downloaded)")

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                row["title"],
                row["artist"],
                row["url"].rsplit("/", 1)[-1],
            ): row
            for row in to_download
        }

        for i, future in enumerate(as_completed(futures), 1):
            track_id, error = future.result()
            if error:
                row = futures[future]
                failed.append(row["url"])
                print(f"  [{i}/{len(to_download)}] FAILED {track_id}: {error}")
            else:
                if i % 50 == 0 or i == len(to_download):
                    print(f"  [{i}/{len(to_download)}] downloaded")

    if failed:
        print(f"\n{len(failed)} songs failed to download")
        with open(os.path.join(DATA_DIR, "failed_urls.txt"), "w") as f:
            f.write("\n".join(failed))
