from engineer_high_level_features import build_high_level_features


def build_human_features(input_csv: str, output_csv: str) -> None:
    build_high_level_features(input_csv, output_csv)


def main() -> None:
    from engineer_high_level_features import main as _main

    _main()


if __name__ == "__main__":
    main()
