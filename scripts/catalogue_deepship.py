#!/usr/bin/env python3
"""Create a simple metadata table for the DeepShip recordings."""

import csv
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATASET_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
OUTPUT_FILE = PROJECT_FOLDER / "data/catalogues/deepship_recordings.csv"


def read_audio_info(audio_file):
    """Read the duration and sample rate with ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=sample_rate",
        "-of",
        "json",
        str(audio_file),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    audio_info = json.loads(result.stdout)

    if len(audio_info["streams"]) != 1:
        raise ValueError(f"Expected one audio stream in {audio_file}")

    duration = float(audio_info["format"]["duration"])
    sample_rate = int(audio_info["streams"][0]["sample_rate"])

    return duration, sample_rate


def read_source_metadata():
    """Read vessel, date, and time values from the DeepShip metafiles."""
    source_metadata = {}

    for class_folder in DATASET_FOLDER.iterdir():
        if not class_folder.is_dir():
            continue

        metadata_files = list(class_folder.glob("*-metafile"))
        if not metadata_files:
            continue

        with open(
            metadata_files[0],
            encoding="utf-8-sig",
            errors="replace",
        ) as metadata_file:
            for metadata_row in csv.reader(metadata_file):
                if len(metadata_row) < 5:
                    continue

                recording_id = metadata_row[0].strip()
                vessel_name = metadata_row[2].strip()
                recording_date = metadata_row[3].strip()
                recording_time = metadata_row[4].strip()

                metadata_key = (class_folder.name, recording_id)

                if metadata_key not in source_metadata:
                    source_metadata[metadata_key] = {
                        "vessel_name": vessel_name,
                        "recording_date": recording_date,
                        "recording_time": recording_time,
                    }

    return source_metadata


def main():
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required. Install FFmpeg and try again.")

    if not DATASET_FOLDER.is_dir():
        raise SystemExit("DeepShip was not found. Run download_deepship.py first.")

    source_metadata = read_source_metadata()
    audio_files = sorted(DATASET_FOLDER.glob("*/*.wav"))
    metadata_rows = []

    if not audio_files:
        raise SystemExit("No WAV files were found. Run download_deepship.py first.")

    for audio_file in audio_files:
        class_name = audio_file.parent.name
        recording_id = audio_file.stem
        duration, sample_rate = read_audio_info(audio_file)

        metadata_key = (class_name, recording_id)
        recording_metadata = source_metadata.get(metadata_key, {})

        vessel_name = recording_metadata.get("vessel_name", "")
        recording_date = recording_metadata.get("recording_date", "")
        recording_time = recording_metadata.get("recording_time", "")

        session_identifier = ""
        if recording_date and recording_time:
            session_identifier = (
                f"{class_name}-{recording_date}-{recording_time}-{recording_id}"
            )

        metadata_rows.append(
            {
                "file": audio_file.relative_to(DATASET_FOLDER).as_posix(),
                "class": class_name,
                "duration_seconds": round(duration, 3),
                "sample_rate_hz": sample_rate,
                "source_recording": recording_id,
                "vessel_name": vessel_name,
                "session_identifier": session_identifier,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as output_file:
        column_names = [
            "file",
            "class",
            "duration_seconds",
            "sample_rate_hz",
            "source_recording",
            "vessel_name",
            "session_identifier",
        ]
        csv_writer = csv.DictWriter(
            output_file,
            fieldnames=column_names,
            lineterminator="\n",
        )
        csv_writer.writeheader()
        csv_writer.writerows(metadata_rows)

    print(f"Saved {len(metadata_rows)} recordings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
