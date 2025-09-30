"""Shared utilities."""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, TypeVar

LOGGER_NAME = "cwa"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging once."""
    if logging.getLogger(LOGGER_NAME).handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def slugify(value: str) -> str:
    """Return a filesystem-safe slug for the provided string."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "untitled"


T = TypeVar("T")


def batched(sequence: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield batches of size ``batch_size`` from ``sequence``."""
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]


def backoff_sleep(attempt: int, base_delay: float = 0.5, factor: float = 2.0, cap: float = 30.0) -> None:
    """Sleep for an exponential backoff duration based on the attempt index."""
    delay = min(cap, base_delay * (factor**attempt))
    time.sleep(delay)


def safe_div(numerator: float, denominator: float) -> float:
    """Safely divide and handle zero denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def chess_result_to_score(result: str, player_color: str) -> float:
    """Convert PGN result notation to a numeric score from the player's perspective."""
    outcome_map = {
        ("1-0", "white"): 1.0,
        ("1-0", "black"): 0.0,
        ("0-1", "white"): 0.0,
        ("0-1", "black"): 1.0,
        ("1/2-1/2", "white"): 0.5,
        ("1/2-1/2", "black"): 0.5,
    }
    return outcome_map.get((result, player_color.lower()), 0.0)


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory of ``path`` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def clamp(value: float, floor: float, ceiling: float) -> float:
    """Clamp a value between floor and ceiling."""
    return max(floor, min(ceiling, value))


def pearsonr_safe(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Compute Pearson correlation guarding against degenerate cases."""
    if len(x) != len(y) or len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


__all__ = [
    "configure_logging",
    "get_logger",
    "slugify",
    "batched",
    "backoff_sleep",
    "safe_div",
    "chess_result_to_score",
    "ensure_parent",
    "clamp",
    "pearsonr_safe",
]
