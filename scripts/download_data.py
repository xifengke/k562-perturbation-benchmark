"""Download and verify the Norman 2019 scPerturb AnnData file.

The downloader uses only Python's standard library. It writes to a temporary
``.part`` file, attempts HTTP Range resume, verifies size and MD5, and only
then promotes the file to its final name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


FILENAME = "NormanWeissman2019_filtered.h5ad"
URL = (
    "https://zenodo.org/records/13350497/files/"
    "NormanWeissman2019_filtered.h5ad?download=1"
)
EXPECTED_BYTES = 698_680_199
EXPECTED_MD5 = "c870e6967d91c017d9da827bab183cd6"
CHUNK_SIZE = 8 * 1024 * 1024


def md5sum(path: Path) -> str:
    """Return the hexadecimal MD5 digest of a file."""
    digest = hashlib.md5()  # noqa: S324 - required to verify the published checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> tuple[bool, str]:
    """Validate file size and checksum, returning a readable status message."""
    actual_bytes = path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        return False, f"size mismatch: expected {EXPECTED_BYTES}, got {actual_bytes}"

    actual_md5 = md5sum(path)
    if actual_md5.lower() != EXPECTED_MD5:
        return False, f"MD5 mismatch: expected {EXPECTED_MD5}, got {actual_md5}"
    return True, f"verified {actual_bytes} bytes, MD5 {actual_md5}"


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def show_progress(downloaded: int, started_at: float) -> None:
    """Print one in-place progress line."""
    elapsed = max(time.monotonic() - started_at, 1e-6)
    percent = min(downloaded / EXPECTED_BYTES * 100, 100.0)
    speed = downloaded / elapsed
    print(
        f"\rDownloaded {format_bytes(downloaded)} / "
        f"{format_bytes(EXPECTED_BYTES)} ({percent:5.1f}%) "
        f"at {format_bytes(int(speed))}/s",
        end="",
        flush=True,
    )


def download(part_path: Path, timeout: int) -> None:
    """Download into part_path and resume it when the server supports Range."""
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing > EXPECTED_BYTES:
        raise RuntimeError(
            f"partial file is larger than expected ({existing} > {EXPECTED_BYTES}); "
            "rerun with --force"
        )

    headers = {"User-Agent": "mini-vc-data-downloader/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {format_bytes(existing)}: {part_path}")
    else:
        print(f"Downloading: {URL}")

    request = urllib.request.Request(URL, headers=headers)
    started_at = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range")

            if existing and status == 206 and content_range:
                expected_prefix = f"bytes {existing}-"
                if not content_range.startswith(expected_prefix):
                    raise RuntimeError(
                        f"unexpected Content-Range {content_range!r}; "
                        f"expected prefix {expected_prefix!r}"
                    )
                mode = "ab"
                downloaded = existing
            elif status == 200:
                if existing:
                    print("Server ignored Range; restarting the partial download.")
                mode = "wb"
                downloaded = 0
            else:
                raise RuntimeError(f"unexpected HTTP status {status}")

            with part_path.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    show_progress(downloaded, started_at)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"download failed: {exc}. The partial file is kept for a later retry."
        ) from exc

    print()


def write_manifest(output_dir: Path, final_path: Path) -> Path:
    """Write machine-readable provenance after successful validation."""
    manifest_path = output_dir / "NormanWeissman2019_filtered.download.json"
    manifest = {
        "dataset": "NormanWeissman2019",
        "file": final_path.name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "scPerturb RNA collection on Zenodo",
        "scperturb_version": "1.4",
        "zenodo_record": "13350497",
        "zenodo_doi": "10.5281/zenodo.13350497",
        "url": URL,
        "geo_accession": "GSE133344",
        "paper_doi": "10.1126/science.aax4438",
        "size_bytes": final_path.stat().st_size,
        "md5": md5sum(final_path),
        "dataset_license": "No explicit dataset-specific license confirmed",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download and verify the Norman 2019 scPerturb H5AD file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="destination directory (default: <repository>/data/raw)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="network timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove an existing final/partial file and download again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / FILENAME
    part_path = output_dir / f"{FILENAME}.part"
    manifest_path = output_dir / "NormanWeissman2019_filtered.download.json"

    if args.force:
        final_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    if final_path.exists():
        valid, message = validate(final_path)
        if not valid:
            print(f"ERROR: existing file failed validation: {message}", file=sys.stderr)
            print("Rerun with --force to replace it.", file=sys.stderr)
            return 1
        if not manifest_path.exists():
            manifest_path = write_manifest(output_dir, final_path)
        print(f"Already downloaded; {message}")
        print(f"Provenance: {manifest_path}")
        return 0

    try:
        download(part_path, args.timeout)
        valid, message = validate(part_path)
        if not valid:
            raise RuntimeError(message)
        os.replace(part_path, final_path)
        manifest_path = write_manifest(output_dir, final_path)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {final_path}")
    print(f"Validation: {message}")
    print(f"Provenance: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
