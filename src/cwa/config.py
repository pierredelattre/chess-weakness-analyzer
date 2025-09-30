"""Application configuration using Pydantic settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Configuration values loaded from environment or .env file."""

    chesscom_username: Optional[str] = Field(default=None, alias="CHESSCOM_USERNAME")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    stockfish_path: Optional[Path] = Field(default=None, alias="STOCKFISH_PATH")

    max_months: Optional[int] = Field(default=None, alias="MAX_MONTHS")
    engine_depth: int = Field(default=14, alias="ENGINE_DEPTH")
    engine_nodes: Optional[int] = Field(default=None, alias="ENGINE_NODES")
    blunder_cp: int = Field(default=-200, alias="BLUNDER_CP")
    win_thresh_cp: int = Field(default=200, alias="WIN_THRESH_CP")
    non_conversion_window: int = Field(default=6, alias="NON_CONVERSION_WINDOW")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "populate_by_name": True,
    }

    def raw_dir(self) -> Path:
        """Return path to the raw data directory."""
        return self.data_dir / "raw"

    def processed_dir(self) -> Path:
        """Return path to the processed data directory."""
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings."""
    settings = AppSettings()
    ensure_data_directories(settings)
    return settings


def ensure_data_directories(settings: AppSettings) -> None:
    """Ensure that required data directories exist."""
    settings.raw_dir().mkdir(parents=True, exist_ok=True)
    settings.processed_dir().mkdir(parents=True, exist_ok=True)


__all__ = ["AppSettings", "get_settings", "ensure_data_directories"]
