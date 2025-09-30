"""Chess.com ingestion logic with caching and retries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests import Session
from tqdm import tqdm

from cwa.config import AppSettings
from cwa.utils import backoff_sleep, ensure_parent, get_logger, slugify

LOGGER = get_logger("data.ingest")
ARCHIVES_ENDPOINT = "https://api.chess.com/pub/player/{username}/games/archives"
DEFAULT_HEADERS = {"User-Agent": "ChessWeaknessAnalyzer/0.1"}


@dataclass
class FetchResult:
    """Represents a fetched archive payload."""

    url: str
    path: Path
    payload: Dict


def list_archives(username: str, session: Optional[Session] = None) -> List[str]:
    """Return the list of archive URLs available for a Chess.com user."""
    session = session or requests.Session()
    response = session.get(ARCHIVES_ENDPOINT.format(username=username.lower()), headers=DEFAULT_HEADERS)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to list archives for {username} (status {response.status_code}): {response.text}"
        )
    data = response.json()
    archives = data.get("archives", [])
    LOGGER.info("Found %s archives for %s", len(archives), username)
    return archives


def fetch_month(url: str, session: Optional[Session] = None, max_retries: int = 4) -> Dict:
    """Fetch a single monthly archive JSON with retry/backoff."""
    session = session or requests.Session()
    attempt = 0
    while True:
        response = session.get(url, headers=DEFAULT_HEADERS)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            LOGGER.warning("Fetch %s failed with %s, retrying...", url, response.status_code)
            attempt += 1
            backoff_sleep(attempt)
            continue
        raise RuntimeError(f"Failed to fetch {url} (status {response.status_code}): {response.text}")


def extract_year_month(identifier: str) -> Tuple[int, int]:
    """Extract (year, month) from an archive URL or cached filename."""
    match = re.search(r"(20\d{2})[/_-]?([01]\d)", identifier)
    if not match:
        raise ValueError(f"Unable to extract year/month from: {identifier}")
    year = int(match.group(1))
    month = int(match.group(2))
    return year, month


def raw_month_path(settings: AppSettings, username: str, year: int, month: int) -> Path:
    """Compute cache path for a monthly archive."""
    slug = slugify(username)
    return settings.raw_dir() / f"chesscom_{slug}_{year}_{month:02d}.json"


def save_raw(path: Path, payload: Dict) -> None:
    """Persist raw payload to disk."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def to_pgns(payload: Dict) -> List[str]:
    """Extract PGN strings from the monthly payload."""
    games = payload.get("games", [])
    return [game["pgn"] for game in games if "pgn" in game]


def download_archives(
    username: str,
    settings: AppSettings,
    max_months: Optional[int] = None,
    force: bool = False,
    session: Optional[Session] = None,
) -> List[FetchResult]:
    """Download archives for ``username`` respecting cache and limits."""
    session = session or requests.Session()
    archive_urls = list_archives(username, session=session)
    if not archive_urls:
        LOGGER.warning("No archives found for %s", username)
        return []

    archive_urls.sort()
    if max_months is not None:
        archive_urls = archive_urls[-max_months:]

    results: List[FetchResult] = []
    for url in tqdm(archive_urls, desc="Downloading archives", unit="month"):
        year, month = extract_year_month(url)
        cache_path = raw_month_path(settings, username, year, month)
        if cache_path.exists() and not force:
            LOGGER.info("Using cached archive %s", cache_path.name)
            with cache_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            results.append(FetchResult(url=url, path=cache_path, payload=payload))
            continue

        payload = fetch_month(url, session=session)
        save_raw(cache_path, payload)
        LOGGER.info("Saved archive %s", cache_path.name)
        results.append(FetchResult(url=url, path=cache_path, payload=payload))
    return results


def load_cached_archives(username: str, settings: AppSettings) -> List[FetchResult]:
    """Load all cached archives for username from disk."""
    slug = slugify(username)
    pattern = f"chesscom_{slug}_*.json"
    results: List[FetchResult] = []
    for path in sorted(settings.raw_dir().glob(pattern)):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        year, month = extract_year_month(path.name)
        url = f"https://api.chess.com/pub/player/{username}/games/{year:04d}/{month:02d}"
        results.append(FetchResult(url=url, path=path, payload=payload))
    return results


__all__ = [
    "FetchResult",
    "download_archives",
    "fetch_month",
    "list_archives",
    "load_cached_archives",
    "raw_month_path",
    "save_raw",
    "to_pgns",
]
