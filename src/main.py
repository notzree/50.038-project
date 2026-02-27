import polars as pl

from download import download_dataset, unify_title_url_mappings


def main():
    csv_path = download_dataset()

    # Lazy scan - no full load into memory
    unified_lf = unify_title_url_mappings(csv_path)

    # Collect stats to verify dedup worked
    stats = unified_lf.select(
        pl.col("url").n_unique().alias("unique_urls"),
        pl.struct("title", "artist").n_unique().alias("unique_song_artist_pairs"),
    ).collect()
    print(f"Unique (title, artist) pairs: {stats['unique_song_artist_pairs'].item()}")
    print(f"Unique URLs:                  {stats['unique_urls'].item()}")

    # Diagnostic: for each URL with multiple (title, artist) pairs, show all variants
    lf = pl.scan_csv(csv_path)
    # Get all distinct (url, title, artist) combos
    all_combos = lf.select("url", "title", "artist").unique().collect()
    # Find URLs that appear with more than one (title, artist) pair
    url_counts = all_combos.group_by("url").agg(pl.len().alias("n_variants"))
    dupe_urls = url_counts.filter(pl.col("n_variants") > 1)
    # Join back to get all variants for those URLs
    dupe_details = all_combos.join(dupe_urls.select("url"), on="url", how="inner").sort(
        "url", "title", "artist"
    )
    dupe_details.write_csv("data/dupe_diagnostics.csv")
    print(
        f"\nWrote {len(dupe_details)} rows for {len(dupe_urls)} duplicate URLs to data/dupe_diagnostics.csv"
    )


if __name__ == "__main__":
    main()
