"""Download the official Reactome pathway GMT with a reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
VERSION_URL = "https://reactome.org/ContentService/data/database/version"
OUTPUT = ROOT / "data/external/reactome"
ARCHIVE = OUTPUT / "ReactomePathways.gmt.zip"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "download_manifest.json"
    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
        request = urllib.request.Request(URL, headers={"User-Agent": "miniVC-K562/0.3"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            partial.replace(ARCHIVE)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt Reactome ZIP member: {bad_member}")
        members = archive.namelist()
        if not any(name.endswith(".gmt") for name in members):
            raise ValueError("Reactome archive has no GMT member")

    archive_sha256 = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if existing_manifest is not None:
        if existing_manifest.get("sha256") != archive_sha256:
            raise ValueError("Existing Reactome archive differs from its download manifest")
        print(json.dumps(existing_manifest, indent=2, ensure_ascii=False))
        return

    try:
        with urllib.request.urlopen(VERSION_URL, timeout=15) as response:
            version = response.read().decode("utf-8").strip()
    except Exception:
        version = None

    manifest = {
        "source_url": URL,
        "version_url": VERSION_URL,
        "reactome_version_at_download": version,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_bytes": ARCHIVE.stat().st_size,
        "sha256": archive_sha256,
        "members": members,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
