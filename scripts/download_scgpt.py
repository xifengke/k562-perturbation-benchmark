"""Download and verify the official scGPT whole-human checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checkpoints" / "scgpt_whole_human"
URL = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing"
EXPECTED_SHA256 = {
    "best_model.pt": "6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922",
    "vocab.json": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if all((OUTPUT / name).is_file() and sha256(OUTPUT / name) == expected for name, expected in EXPECTED_SHA256.items()) and (OUTPUT / "args.json").is_file():
        print(f"Verified existing scGPT checkpoint: {OUTPUT}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=URL, output=str(OUTPUT), quiet=False, remaining_ok=True)
    for name, expected in EXPECTED_SHA256.items():
        path = OUTPUT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"missing or unexpected official checkpoint file: {path}")
    if not (OUTPUT / "args.json").is_file():
        raise ValueError("scGPT args.json is missing")
    print(f"Verified scGPT checkpoint: {OUTPUT}")


if __name__ == "__main__":
    main()
