"""Helpers for reading and writing processed datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from cwa.config import AppSettings
from cwa.utils import ensure_parent


def processed_path(settings: AppSettings, username: str, name: str) -> Path:
    """Return canonical path for a processed dataset."""
    safe_username = username.lower().replace(" ", "_")
    return settings.processed_dir() / f"{safe_username}_{name}.parquet"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet."""
    ensure_parent(path)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Load Parquet file into DataFrame."""
    return pd.read_parquet(path)


def append_parquet(df: pd.DataFrame, path: Path, dedupe_keys: Optional[Iterable[str]] = None) -> None:
    """Append rows to a Parquet file while de-duplicating on ``dedupe_keys``."""
    if path.exists():
        existing = read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    if dedupe_keys:
        combined = combined.drop_duplicates(subset=list(dedupe_keys))

    write_parquet(combined, path)


__all__ = ["processed_path", "write_parquet", "read_parquet", "append_parquet"]
