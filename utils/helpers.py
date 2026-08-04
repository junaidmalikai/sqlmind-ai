"""General helper utilities."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"


def slugify(value: str, max_len: int = 48) -> str:
    """Create a filesystem-safe slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return (text or "export")[:max_len]


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
