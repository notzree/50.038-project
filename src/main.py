import argparse

from dotenv import load_dotenv

from download import download_dataset, get_mp3s_for_dataset, unify_title_url_mappings

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of songs to download",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    print("Starting dataset + audio download pipeline")
    print(f"Download limit: {args.limit if args.limit is not None else 'none'}")
    print(f"Max workers: {args.max_workers}")

    csv_path = download_dataset()
    print(f"Using charts dataset at: {csv_path}")

    # Lazy scan - no full load into memory
    unified_lf = unify_title_url_mappings(csv_path)
    get_mp3s_for_dataset(unified_lf, max_workers=args.max_workers, limit=args.limit)
    print("Download pipeline complete")


if __name__ == "__main__":
    main()
