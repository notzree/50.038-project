#!/usr/bin/env python3
"""One-time script to deduplicate audio_features.csv, keeping the last occurrence of each track_id."""

import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/data/audio_features.csv"

df = pd.read_csv(path, on_bad_lines="warn")
print(f"Before: {len(df)} rows, {df['track_id'].nunique()} unique track_ids")

df = df.drop_duplicates(subset=["track_id"], keep="last")
print(f"After:  {len(df)} rows")

df.to_csv(path, index=False)
print(f"Wrote {path}")
