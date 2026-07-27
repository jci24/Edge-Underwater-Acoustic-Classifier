#!/usr/bin/env python3
"""Catalogue downloaded DeepShip WAV files and their source metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("data/raw/deepship")
DEFAULT_OUTPUT = Path("data/catalogues")
FIELDS = [
    "relative_path",
    "class",
    "recording_id",
    "filename",
    "bytes",
    "sha256",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "sample_width_bits",
    "frame_count",
    "audio_format",
    "vessel_name",
    "recording_date",
    "recording_time",
    "source_field_2",
    "source_field_3",
    "source_field_4",
    "source_field_5",
    "source_field_6",
    "source_field_7",
    "source_repository",
    "source_commit",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_audio(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        (
            "stream=codec_name,codec_long_name,sample_rate,channels,"
            "bits_per_sample,duration_ts,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one audio stream in {path}, found {len(streams)}")
    return streams[0]


def load_metadata(root: Path, class_name: str) -> dict[str, list[str]]:
    metafiles = list((root / class_name).glob("*-metafile"))
    if len(metafiles) != 1:
        return {}
    rows: dict[str, list[str]] = {}
    with metafiles[0].open(encoding="utf-8-sig", errors="replace", newline="") as source:
        for fields in csv.reader(source):
            cleaned = [field.strip() for field in fields]
            if cleaned and cleaned[0] and cleaned[0] not in rows:
                rows[cleaned[0]] = cleaned
    return rows


def metadata_fields(fields: list[str]) -> dict[str, str]:
    padded = (fields + [""] * 7)[:7]
    return {
        "vessel_name": padded[2],
        "recording_date": padded[3],
        "recording_time": padded[4],
        **{f"source_field_{index}": padded[index - 1] for index in range(2, 8)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.input / "source_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"missing {manifest_path}; run download_deepship.py first")
    if shutil.which("ffprobe") is None:
        parser.error("ffprobe is required; install FFmpeg and try again")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wav_files = sorted(args.input.glob("*/*.wav"), key=lambda item: item.as_posix().lower())
    if not wav_files:
        parser.error(f"no WAV files found below {args.input}")

    metadata_by_class = {
        directory.name: load_metadata(args.input, directory.name)
        for directory in args.input.iterdir()
        if directory.is_dir()
    }
    rows = []
    for path in wav_files:
        class_name = path.parent.name
        recording_id = path.stem
        audio = probe_audio(path)
        channels = int(audio["channels"])
        sample_width_bits = int(audio["bits_per_sample"])
        sample_rate = int(audio["sample_rate"])
        duration = float(audio["duration"])
        frame_count = int(audio.get("duration_ts", round(duration * sample_rate)))
        row = {
            "relative_path": path.relative_to(args.input).as_posix(),
            "class": class_name,
            "recording_id": recording_id,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "duration_seconds": f"{duration:.6f}",
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bits": sample_width_bits,
            "frame_count": frame_count,
            "audio_format": audio.get("codec_long_name", audio["codec_name"]),
            **metadata_fields(metadata_by_class.get(class_name, {}).get(recording_id, [])),
            "source_repository": manifest["repository"],
            "source_commit": manifest["source_commit"],
        }
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    catalogue_path = args.output / "deepship_recordings.csv"
    with catalogue_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    class_summary = defaultdict(lambda: {"recordings": 0, "bytes": 0, "duration_seconds": 0.0})
    for row in rows:
        summary = class_summary[row["class"]]
        summary["recordings"] += 1
        summary["bytes"] += int(row["bytes"])
        summary["duration_seconds"] += float(row["duration_seconds"])
    summary = {
        "dataset": "DeepShip public GitHub subset",
        "source_repository": manifest["repository"],
        "source_commit": manifest["source_commit"],
        "recordings": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "duration_seconds": round(sum(float(row["duration_seconds"]) for row in rows), 6),
        "classes": {
            name: {
                **values,
                "duration_seconds": round(values["duration_seconds"], 6),
            }
            for name, values in sorted(class_summary.items())
        },
    }
    summary_path = args.output / "deepship_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Catalogued {len(rows)} recordings in {catalogue_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
