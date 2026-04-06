import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import kagglehub
import pandas as pd

from download import DATA_DIR, SONGS_DIR, _download_one

NONVIRAL_META_CSV = os.path.join(DATA_DIR, "nonviral_track_ids.csv")


def download_nonviral_dataset() -> str:
    """Download the Kaggle Spotify tracks dataset. Returns the CSV path."""
    path = kagglehub.dataset_download(
        "maharshipandya/-spotify-tracks-dataset",
        path="dataset.csv",
        output_dir=DATA_DIR,
    )
    return path


def get_nonviral_mp3s(
    limit: int | None = None,
    max_workers: int = 12,
    popularity_threshold: int = 25,
) -> None:
    """Download MP3s for low-popularity tracks from the Kaggle Spotify dataset.

    Filters to tracks with popularity < threshold, skips already-downloaded
    track IDs, downloads in parallel, and writes nonviral_track_ids.csv.
    """
    os.makedirs(SONGS_DIR, exist_ok=True)

    csv_path = download_nonviral_dataset()
    df = pd.read_csv(csv_path)

    # Filter to low-popularity tracks
    df = df[df["popularity"] < popularity_threshold].copy()

    # Drop rows without a track_id, track_name, or artists
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    df = df.drop_duplicates(subset=["track_id"])

    # Get existing track IDs from songs dir
    existing = {f.removesuffix(".mp3") for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")}

    # Also load previously recorded nonviral IDs so we don't re-attempt known failures
    recorded_nonviral: set[str] = set()
    if os.path.exists(NONVIRAL_META_CSV):
        recorded_nonviral = set(pd.read_csv(NONVIRAL_META_CSV)["track_id"].astype(str))

    to_download = [
        row
        for _, row in df.iterrows()
        if str(row["track_id"]) not in existing and str(row["track_id"]) not in recorded_nonviral
    ]

    if limit is not None:
        to_download = to_download[:limit]

    print(
        f"Non-viral candidates: {len(df)} | already downloaded: {len(existing)} | "
        f"to download: {len(to_download)}"
    )

    successful: list[str] = list(recorded_nonviral)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                str(row["track_name"]),
                str(row["artists"]),
                str(row["track_id"]),
            ): row
            for row in to_download
        }

        for i, future in enumerate(as_completed(futures), 1):
            track_id, error = future.result()
            if error:
                print(f"  [{i}/{len(to_download)}] FAILED {track_id}: {error}")
            else:
                successful.append(track_id)
                if i % 50 == 0 or i == len(to_download):
                    print(f"  [{i}/{len(to_download)}] downloaded")

    # Write metadata CSV
    meta = pd.DataFrame({"track_id": successful, "global_nonviral": 1})
    meta.to_csv(NONVIRAL_META_CSV, index=False)
    print(f"Wrote {len(meta)} non-viral track IDs to {NONVIRAL_META_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download non-viral songs from Kaggle")
    parser.add_argument("--limit", type=int, default=None, help="Max songs to download")
    parser.add_argument("--max-workers", type=int, default=12, help="Download workers")
    parser.add_argument(
        "--popularity-threshold",
        type=int,
        default=25,
        help="Max popularity score to qualify as non-viral (default: 25)",
    )
    args = parser.parse_args()

    get_nonviral_mp3s(
        limit=args.limit,
        max_workers=args.max_workers,
        popularity_threshold=args.popularity_threshold,
    )


if __name__ == "__main__":
    main()
