import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from extract_features import extract_basic_features


def pick_audio_file_from_finder() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter is not available in this Python build. "
            "Use --audio <path> instead of --pick-file."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not selected:
        raise ValueError("No file selected in Finder dialog")
    return selected


def normalize_user_path(raw: str) -> str:
    cleaned = raw.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def predict_single(audio_path: str, region: str, model_path: str) -> dict:
    print(f"Loading model: {model_path}")
    clf = joblib.load(model_path)

    print(f"Extracting features from: {audio_path}")
    features = extract_basic_features(audio_path)
    row = {"region": region, **features}
    X = pd.DataFrame([row])

    pred = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0, 1])

    return {
        "audio_path": audio_path,
        "region": region,
        "prediction": pred,
        "probability_appears_in_region": prob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict if a single audio file appears in a region"
    )
    audio_group = parser.add_mutually_exclusive_group(required=False)
    audio_group.add_argument(
        "--audio", help="Path to audio file (mp3/wav/m4a/flac/ogg)"
    )
    audio_group.add_argument(
        "--pick-file",
        action="store_true",
        help="Open Finder dialog to select audio file",
    )
    parser.add_argument("--region", required=False, help="Region name used in training")
    parser.add_argument(
        "--model",
        default="src/data/mvp_model.joblib",
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write prediction result JSON",
    )
    args = parser.parse_args()

    audio_input = args.audio
    if args.pick_file:
        print("Opening Finder file picker...")
        audio_input = pick_audio_file_from_finder()

    if not audio_input:
        audio_input = input("Enter audio file path: ").strip()
    audio_input = normalize_user_path(audio_input)

    region = args.region
    if not region:
        region = input("Enter region (e.g., Singapore): ").strip()
    region = region.strip().strip("\"'")

    audio_path = Path(audio_input)
    model_path = Path(args.model)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not region:
        raise ValueError("Region is required")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    result = predict_single(str(audio_path), region, str(model_path))

    print("Prediction complete:")
    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Wrote prediction JSON to {out}")


if __name__ == "__main__":
    main()
