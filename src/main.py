from dotenv import load_dotenv

from download import download_dataset, get_mp3s_for_dataset, unify_title_url_mappings

load_dotenv()


def main():
    csv_path = download_dataset()

    # Lazy scan - no full load into memory
    unified_lf = unify_title_url_mappings(csv_path)
    get_mp3s_for_dataset(unified_lf)


if __name__ == "__main__":
    main()
