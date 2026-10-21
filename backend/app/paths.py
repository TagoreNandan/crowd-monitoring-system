"""Filesystem locations anchored to the repository root.

Everything is built with ``pathlib`` from this file's own location, so paths
resolve the same on Windows and POSIX no matter which directory the server
was launched from (a relative ``Path("data/...")`` would silently depend on
the current working directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LIVE_RECORDINGS_DIR = DATA_DIR / "live_recordings"
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_SAMPLE_VIDEO = EXAMPLES_DIR / "sample_crowd.mp4"


def find_local_file(uri: Optional[str]) -> Optional[Path]:
    """`uri` as an existing local file (as given, then relative to the repo
    root), or None -- e.g. for S3 keys and stream URLs."""
    if not uri or not uri.strip() or "://" in uri:
        return None
    candidate = Path(uri.strip())
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute() and (REPO_ROOT / candidate).is_file():
        return REPO_ROOT / candidate
    return None


def is_within(path: Path, *roots: Path) -> bool:
    """True if `path` resolves inside any of `roots` (case-insensitive on
    Windows, and immune to the "C:\\repo" vs "C:\\repo2" prefix trap)."""
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)
