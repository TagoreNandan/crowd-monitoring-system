"""Pretrained weight registry and downloader.

Weights for the wrapped models are published as plain GitHub release
assets by the LWCC project (https://github.com/tersekmatija/lwcc_weights),
which mirrors the original DM-Count / CSRNet checkpoints. That means they
can be fetched with a normal HTTPS GET -- no Google Drive / Baidu / OneDrive
client and no manual browser download required, which matters because a
number of the original crowd-counting papers (including FIDTM) only
publish their checkpoints on those services.

Downloaded files are cached under WEIGHTS_DIR and never re-downloaded once
present.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = Path(os.environ.get("CROWDMON_WEIGHTS_DIR", REPO_ROOT / "weights"))

_RELEASE_BASE = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1"

# model_name -> {dataset_weights: file name on the release}
AVAILABLE_WEIGHTS = {
    "DM-Count": {"SHA": "DM-Count_SHA.pth", "SHB": "DM-Count_SHB.pth", "QNRF": "DM-Count_QNRF.pth"},
    "CSRNet": {"SHA": "CSRNet_SHA.pth", "SHB": "CSRNet_SHB.pth"},
}

# Rough guidance on which pretrained weights suit which crowd density.
WEIGHTS_NOTES = {
    "SHA": "ShanghaiTech Part A: dense, mixed indoor/outdoor crowds.",
    "SHB": "ShanghaiTech Part B: sparser street-level crowds.",
    "QNRF": "UCF-QNRF: very large range (~50 to ~12,000 people/image); best default for unknown/extreme density.",
}


def weight_path(model_name: str, model_weights: str) -> Path:
    if model_name not in AVAILABLE_WEIGHTS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(AVAILABLE_WEIGHTS)}")
    if model_weights not in AVAILABLE_WEIGHTS[model_name]:
        raise ValueError(
            f"Weights '{model_weights}' not available for {model_name}. "
            f"Available: {list(AVAILABLE_WEIGHTS[model_name])}"
        )
    file_name = AVAILABLE_WEIGHTS[model_name][model_weights]
    return WEIGHTS_DIR / file_name


def ensure_weights(model_name: str, model_weights: str, progress: bool = True) -> Path:
    """Return a local path to the requested checkpoint, downloading it if needed."""
    path = weight_path(model_name, model_weights)
    if path.exists() and path.stat().st_size > 0:
        return path

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = path.name
    url = f"{_RELEASE_BASE}/{file_name}"
    tmp_path = path.with_suffix(path.suffix + ".part")

    logger.info("Downloading pretrained weights %s from %s", file_name, url)
    req = Request(url, headers={"User-Agent": "crowd-monitoring-system/1.0"})
    with urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 256
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    pct = downloaded / total * 100
                    print(f"\r  {file_name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        if progress and total:
            print()

    if total and downloaded != total:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Incomplete download of {file_name}: got {downloaded} of {total} bytes. "
            f"Try again -- the partial file was removed."
        )

    tmp_path.rename(path)
    logger.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def is_downloaded(model_name: str, model_weights: str) -> bool:
    path = weight_path(model_name, model_weights)
    return path.exists() and path.stat().st_size > 0
