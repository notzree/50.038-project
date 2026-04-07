"""Standalone genre fetcher via Spotify API.

Runs independently from the main pipeline. Can be executed in parallel with
audio feature extraction since it only needs track IDs, not audio files.

Usage:
    uv run python src/fetch_genres.py --input src/data/audio_manifest.csv
    uv run python src/fetch_genres.py --input src/data/charts.csv --id-column url

Requires SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env or environment.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 50  # Spotify API max per request


def _init_spotify():
    """Initialize spotipy client with client credentials flow."""
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env or environment"
        )

    auth_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    return spotipy.Spotify(
        auth_manager=auth_manager,
        requests_timeout=10,
        retries=5,
        backoff_factor=0.5,
    )


def _extract_track_ids(input_path: str, id_column: str) -> list[str]:
    """Extract unique Spotify track IDs from input CSV."""
    df = pd.read_csv(input_path)

    if id_column not in df.columns:
        raise ValueError(
            f"Column '{id_column}' not found. Available: {list(df.columns)}"
        )

    if id_column == "url":
        track_ids = (
            df["url"]
            .str.split("?").str[0]
            .str.rsplit("/", n=1).str[-1]
            .unique()
            .tolist()
        )
    else:
        track_ids = df[id_column].unique().tolist()

    return [tid for tid in track_ids if tid and isinstance(tid, str)]


def _load_existing(output_path: str) -> set[str]:
    """Load already-fetched track IDs from output CSV."""
    path = Path(output_path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(output_path)
        return set(df["track_id"].tolist())
    except Exception:
        return set()


def _batch_fetch_genres(
    sp, track_ids: list[str], output_path: str
) -> None:
    """Fetch genres for tracks in batches, writing results incrementally."""
    # Open in append mode
    path = Path(output_path)
    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["track_id", "primary_genre", "genre_list"]
        )
        if not file_exists:
            writer.writeheader()

        # Step 1: Batch fetch tracks to get artist IDs
        total = len(track_ids)
        artist_map = {}  # track_id -> [artist_ids]
        all_artist_ids = set()

        for i in range(0, total, BATCH_SIZE):
            batch = track_ids[i : i + BATCH_SIZE]
            print(f"  Fetching track metadata [{i+1}-{min(i+BATCH_SIZE, total)}/{total}]")

            try:
                results = sp.tracks(batch)
                for track in results["tracks"]:
                    if track is None:
                        continue
                    tid = track["id"]
                    artist_ids = [a["id"] for a in track.get("artists", [])]
                    artist_map[tid] = artist_ids
                    all_artist_ids.update(artist_ids)
            except Exception as e:
                print(f"    Error fetching tracks batch: {e}")
                # Respect Retry-After if present
                if hasattr(e, "headers") and "Retry-After" in getattr(e, "headers", {}):
                    wait = int(e.headers["Retry-After"])
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                continue

        # Step 2: Batch fetch artists to get genres
        artist_genres = {}  # artist_id -> [genres]
        artist_list = list(all_artist_ids)

        for i in range(0, len(artist_list), BATCH_SIZE):
            batch = artist_list[i : i + BATCH_SIZE]
            print(f"  Fetching artist genres [{i+1}-{min(i+BATCH_SIZE, len(artist_list))}/{len(artist_list)}]")

            try:
                results = sp.artists(batch)
                for artist in results["artists"]:
                    if artist is None:
                        continue
                    artist_genres[artist["id"]] = artist.get("genres", [])
            except Exception as e:
                print(f"    Error fetching artists batch: {e}")
                if hasattr(e, "headers") and "Retry-After" in getattr(e, "headers", {}):
                    wait = int(e.headers["Retry-After"])
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                continue

        # Step 3: Aggregate genres per track and write
        written = 0
        for tid in track_ids:
            if tid not in artist_map:
                writer.writerow(
                    {
                        "track_id": tid,
                        "primary_genre": "unknown",
                        "genre_list": "",
                    }
                )
                written += 1
                continue

            # Collect all genres from all artists on this track
            all_genres = []
            for aid in artist_map[tid]:
                all_genres.extend(artist_genres.get(aid, []))

            if all_genres:
                # Pick most common genre as primary
                from collections import Counter

                genre_counts = Counter(all_genres)
                primary = genre_counts.most_common(1)[0][0]
                genre_str = "|".join(sorted(set(all_genres)))
            else:
                primary = "unknown"
                genre_str = ""

            writer.writerow(
                {
                    "track_id": tid,
                    "primary_genre": primary,
                    "genre_list": genre_str,
                }
            )
            written += 1

        print(f"  Wrote {written} genre records")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch genres from Spotify API (standalone, runs independently)"
    )
    parser.add_argument(
        "--input",
        default="src/data/audio_manifest.csv",
        help="Input CSV with track IDs",
    )
    parser.add_argument(
        "--id-column",
        default="track_id",
        help="Column name containing track IDs (or 'url' to extract from Spotify URLs)",
    )
    parser.add_argument(
        "--output",
        default="src/data/genre_features.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print(f"Fetching genres from Spotify API")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Extract track IDs
    all_ids = _extract_track_ids(args.input, args.id_column)
    print(f"Found {len(all_ids)} unique track IDs")

    # Skip already-fetched
    done = _load_existing(args.output)
    remaining = [tid for tid in all_ids if tid not in done]
    print(f"Already fetched: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All tracks already fetched, nothing to do")
        return

    # Initialize Spotify client
    sp = _init_spotify()

    # Fetch in batches
    _batch_fetch_genres(sp, remaining, args.output)

    # Summary
    final = pd.read_csv(args.output)
    n_with_genre = (final["primary_genre"] != "unknown").sum()
    print(f"Done. {n_with_genre}/{len(final)} tracks have genre data")


if __name__ == "__main__":
    main()
