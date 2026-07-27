#!/usr/bin/env python3
"""Download the publicly available portion of DeepShip from GitHub."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPOSITORY = "irfankamboh/DeepShip"
DEFAULT_OUTPUT = Path("data/raw/deepship")
USER_AGENT = "edge-underwater-acoustic-classifier/1.0"


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def wanted(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.lower().endswith(".wav") or name.endswith("-metafile") or name == "README.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ref", default="main", help="Upstream branch, tag, or commit")
    args = parser.parse_args()

    encoded_ref = urllib.parse.quote(args.ref, safe="")
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{encoded_ref}?recursive=1"
    try:
        tree = request_json(tree_url)
    except urllib.error.HTTPError as error:
        print(f"Unable to read upstream manifest: HTTP {error.code}", file=sys.stderr)
        return 1

    if tree.get("truncated"):
        print("GitHub returned a truncated tree; refusing an incomplete download.", file=sys.stderr)
        return 1

    entries = [
        item
        for item in tree.get("tree", [])
        if item.get("type") == "blob" and wanted(item["path"])
    ]
    if not entries:
        print("No DeepShip source files found in the upstream tree.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    for index, item in enumerate(entries, start=1):
        destination = args.output / PurePosixPath(item["path"])
        expected_size = int(item.get("size", 0))
        if destination.is_file() and destination.stat().st_size == expected_size:
            status = "skip"
        else:
            quoted_path = urllib.parse.quote(item["path"])
            raw_url = (
                f"https://raw.githubusercontent.com/{REPOSITORY}/"
                f"{encoded_ref}/{quoted_path}"
            )
            download(raw_url, destination)
            if destination.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch after downloading {item['path']}")
            status = "download"
        print(f"[{index:02d}/{total:02d}] {status:8s} {item['path']}")

    manifest = {
        "repository": f"https://github.com/{REPOSITORY}",
        "requested_ref": args.ref,
        "source_commit": tree["sha"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": item["path"], "bytes": item.get("size"), "git_blob_sha": item["sha"]}
            for item in entries
        ],
    }
    manifest_path = args.output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    wav_entries = [item for item in entries if item["path"].lower().endswith(".wav")]
    wav_bytes = sum(int(item.get("size", 0)) for item in wav_entries)
    print(f"Ready: {len(wav_entries)} WAV files ({wav_bytes:,} bytes)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
