"""Build and validate the versioned handcrafted feature table."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Callable

import torch
from torchcodec.decoders import AudioDecoder

from .features import HandcraftedFeatureConfig, HandcraftedFeatureExtractor
from .manifest import MANIFEST_COLUMNS, read_csv_rows
from .preprocessing import PreprocessingConfig


TRACE_COLUMNS = [*MANIFEST_COLUMNS, "feature_config_hash"]


class CachedIntervalDecoder:
    """Reuse one TorchCodec decoder while reading consecutive source windows."""

    def __init__(self, audio_root: Path) -> None:
        self.audio_root = audio_root
        self.current_path: Path | None = None
        self.decoder: AudioDecoder | None = None

    def decode(self, row: dict[str, str]) -> tuple[torch.Tensor, int]:
        audio_path = self.audio_root / row["source_file"]
        if audio_path != self.current_path:
            self.decoder = AudioDecoder(audio_path)
            self.current_path = audio_path
        if self.decoder is None:
            raise RuntimeError(f"Could not decode {audio_path}.")
        samples = self.decoder.get_samples_played_in_range(
            start_seconds=float(row["start_seconds"]),
            stop_seconds=float(row["end_seconds"]),
        )
        return samples.data, samples.sample_rate


def extract_feature_rows(
    manifest_rows: list[dict[str, str]],
    audio_root: Path,
    extractor: HandcraftedFeatureExtractor,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, str | float]]:
    """Extract exactly one traceable row for every manifest window."""

    decoder = CachedIntervalDecoder(audio_root)
    output_rows: list[dict[str, str | float]] = []
    total = len(manifest_rows)

    for index, manifest_row in enumerate(manifest_rows, start=1):
        if (
            manifest_row["config_hash"]
            != extractor.config.preprocessing_config_hash
        ):
            raise ValueError("Manifest and feature preprocessing hashes differ.")
        waveform, sample_rate = decoder.decode(manifest_row)
        features = extractor.extract(waveform, sample_rate)
        output_row: dict[str, str | float] = dict(manifest_row)
        output_row["feature_config_hash"] = extractor.config.config_hash
        output_row.update(features)
        output_rows.append(output_row)
        if progress is not None:
            progress(index, total)

    validate_feature_rows(output_rows, manifest_rows, extractor.config)
    return output_rows


def validate_feature_rows(
    feature_rows: list[dict[str, str | float]],
    manifest_rows: list[dict[str, str]],
    config: HandcraftedFeatureConfig,
) -> None:
    if len(feature_rows) != len(manifest_rows):
        raise ValueError("Feature table and manifest must have the same row count.")
    manifest_by_id = {row["window_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Manifest window IDs must be unique.")

    feature_ids = [str(row["window_id"]) for row in feature_rows]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("Feature table window IDs must be unique.")
    if set(feature_ids) != set(manifest_by_id):
        raise ValueError("Feature table window IDs do not match the manifest.")

    for row in feature_rows:
        manifest_row = manifest_by_id[str(row["window_id"])]
        for column in MANIFEST_COLUMNS:
            if str(row[column]) != str(manifest_row[column]):
                raise ValueError(
                    f"Trace field {column} changed for {row['window_id']}."
                )
        if row["feature_config_hash"] != config.config_hash:
            raise ValueError("Feature row uses a different feature configuration.")
        for name in config.feature_names:
            value = float(row[name])
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Feature {name} is not finite.")


def write_feature_table(
    rows: list[dict[str, str | float]],
    output_path: Path,
    config: HandcraftedFeatureConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*TRACE_COLUMNS, *config.feature_names]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_feature_table(
    manifest_path: Path,
    audio_root: Path,
    preprocessing_path: Path,
    feature_config_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> None:
    preprocessing = PreprocessingConfig.load(preprocessing_path)
    config = HandcraftedFeatureConfig.from_preprocessing(preprocessing)
    extractor = HandcraftedFeatureExtractor(config, preprocessing)
    manifest_rows = read_csv_rows(manifest_path)

    def show_progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"Extracted {done}/{total} windows")

    started = time.perf_counter()
    rows = extract_feature_rows(
        manifest_rows,
        audio_root,
        extractor,
        progress=show_progress,
    )
    elapsed = time.perf_counter() - started
    config.save(feature_config_path)
    write_feature_table(rows, output_path, config)
    metadata = {
        "window_count": len(rows),
        "feature_count": len(config.feature_names),
        "total_extraction_seconds": elapsed,
        "seconds_per_window": elapsed / len(rows),
        "measurement_note": (
            "Warm-cache development-machine extraction; not edge-device performance."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "preprocessing_config_hash": preprocessing.config_hash,
        "feature_config_hash": config.config_hash,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows and 73 features to {output_path}")

