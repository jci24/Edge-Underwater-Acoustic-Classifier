#!/usr/bin/env python3
"""Download the public DeepShip recordings from GitHub."""

import json
import urllib.request
from pathlib import Path


GITHUB_REPOSITORY = "irfankamboh/DeepShip"
GITHUB_BRANCH = "main"
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = PROJECT_FOLDER / "data/raw/deepship"


def is_dataset_file(file_path):
    """Check if a GitHub file belongs to the public dataset."""
    file_name = Path(file_path).name

    return (
        file_path.lower().endswith(".wav")
        or file_name.endswith("-metafile")
        or file_name == "README.txt"
    )


def download_file(github_path, expected_size):
    """Download one file, unless a complete copy already exists."""
    local_file = DOWNLOAD_FOLDER / github_path
    temporary_file = local_file.with_suffix(local_file.suffix + ".part")

    if local_file.is_file() and local_file.stat().st_size == expected_size:
        print(f"Skip: {github_path}")
        return

    local_file.parent.mkdir(parents=True, exist_ok=True)

    download_url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/"
        f"{GITHUB_BRANCH}/{github_path}"
    )

    print(f"Download: {github_path}")
    urllib.request.urlretrieve(download_url, temporary_file)

    if temporary_file.stat().st_size != expected_size:
        temporary_file.unlink()
        raise RuntimeError(f"The download is incomplete: {github_path}")

    temporary_file.replace(local_file)


def main():
    github_api_url = (
        f"https://api.github.com/repos/{GITHUB_REPOSITORY}/"
        f"git/trees/{GITHUB_BRANCH}?recursive=1"
    )
    request = urllib.request.Request(
        github_api_url,
        headers={"User-Agent": "deepship-downloader"},
    )

    with urllib.request.urlopen(request) as response:
        github_data = json.load(response)

    if github_data.get("truncated"):
        raise SystemExit("GitHub returned an incomplete file list. Try again later.")

    dataset_files = []

    for github_file in github_data["tree"]:
        if github_file["type"] == "blob" and is_dataset_file(github_file["path"]):
            dataset_files.append(github_file)

    if not dataset_files:
        raise SystemExit("No DeepShip dataset files were found.")

    audio_file_count = 0
    total_audio_bytes = 0

    for dataset_file in dataset_files:
        file_path = dataset_file["path"]
        file_size = dataset_file["size"]

        download_file(file_path, file_size)

        if file_path.lower().endswith(".wav"):
            audio_file_count += 1
            total_audio_bytes += file_size

    source_commit_file = DOWNLOAD_FOLDER / "source_commit.txt"
    source_commit_file.write_text(github_data["sha"] + "\n", encoding="utf-8")

    print(f"Ready: {audio_file_count} WAV files ({total_audio_bytes:,} bytes)")
    print(f"Source commit: {github_data['sha']}")


if __name__ == "__main__":
    main()
