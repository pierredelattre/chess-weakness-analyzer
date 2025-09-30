"""Stockfish engine management and evaluation helpers."""

from __future__ import annotations

import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import chess
import chess.engine

from cwa.config import AppSettings
from cwa.utils import ensure_parent, get_logger

LOGGER = get_logger("engine.stockfish")

DOWNLOAD_MAP = {
    ("Darwin", "x86_64"): {
        "url": "https://stockfishchess.org/files/stockfish-16-mac.zip",
        "binary": "stockfish-16-mac/MacOS/stockfish-16-mac",
    },
    ("Darwin", "arm64"): {
        "url": "https://stockfishchess.org/files/stockfish-16-mac.zip",
        "binary": "stockfish-16-mac/MacOS/stockfish-16-mac",
    },
    ("Linux", "x86_64"): {
        "url": "https://stockfishchess.org/files/stockfish-ubuntu-x86-64-avx2.zip",
        "binary": "stockfish-ubuntu-x86-64-avx2/stockfish-ubuntu-x86-64-avx2",
    },
}

CACHE_DIR = Path.home() / ".cache" / "cwa" / "stockfish"


class StockfishUnavailableError(RuntimeError):
    """Raised when Stockfish cannot be resolved automatically."""


def resolve_engine_path(settings: AppSettings) -> Path:
    """Find or download a Stockfish binary and return its path."""
    if settings.stockfish_path and settings.stockfish_path.exists():
        return settings.stockfish_path

    env_path = Path(settings.stockfish_path) if settings.stockfish_path else None
    if env_path and env_path.exists():
        return env_path

    which_path = shutil.which("stockfish")
    if which_path:
        return Path(which_path)

    auto_path = attempt_auto_download()
    if auto_path:
        return auto_path

    raise StockfishUnavailableError(
        "Stockfish binary not found. Please install Stockfish and set STOCKFISH_PATH or add it to PATH."
    )


def attempt_auto_download() -> Optional[Path]:
    """Attempt to download a Stockfish binary for the current platform."""
    system = platform.system()
    machine = platform.machine()
    key = (system, machine)
    data = DOWNLOAD_MAP.get(key)
    if not data:
        LOGGER.warning("No Stockfish download mapping for %s/%s", system, machine)
        return None

    url = data["url"]
    binary_rel_path = data["binary"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / Path(url).name
    binary_path = CACHE_DIR / Path(binary_rel_path).name

    if binary_path.exists():
        LOGGER.info("Using cached Stockfish binary at %s", binary_path)
        return binary_path

    try:
        LOGGER.info("Downloading Stockfish from %s", url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Failed to download Stockfish: %s", exc)
        return None

    ensure_parent(archive_path)
    archive_path.write_bytes(response.content)

    try:
        extract_and_install(archive_path, binary_rel_path, binary_path)
    finally:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)

    if binary_path.exists():
        binary_path.chmod(0o755)
        LOGGER.info("Stockfish installed to %s", binary_path)
        return binary_path
    return None


def extract_and_install(archive_path: Path, binary_rel_path: str, target_path: Path) -> None:
    """Extract downloaded archive and move binary into cache."""
    suffix = archive_path.suffix
    if suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                source = Path(tmpdir) / binary_rel_path
                if not source.exists():
                    raise StockfishUnavailableError(
                        f"Stockfish binary {binary_rel_path} missing in archive {archive_path.name}"
                    )
                ensure_parent(target_path)
                shutil.copy2(source, target_path)
    elif suffix in {".tar", ".gz", ".bz2", ".xz"}:
        import tarfile

        with tarfile.open(archive_path, "r:*") as tf:
            with tempfile.TemporaryDirectory() as tmpdir:
                tf.extractall(tmpdir)
                source = Path(tmpdir) / binary_rel_path
                if not source.exists():
                    raise StockfishUnavailableError(
                        f"Stockfish binary {binary_rel_path} missing in archive {archive_path.name}"
                    )
                ensure_parent(target_path)
                shutil.copy2(source, target_path)
    else:
        raise StockfishUnavailableError(f"Unsupported archive format for {archive_path}")


class StockfishEngine:
    """Thin wrapper around python-chess engine usage with caching."""

    def __init__(self, engine_path: Path, depth: int = 14, nodes: Optional[int] = None):
        self.engine_path = engine_path
        self.depth = depth
        self.nodes = nodes
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._cache: Dict[str, Tuple[int, Optional[str]]] = {}

    def __enter__(self) -> "StockfishEngine":
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def analyse_fen(self, fen: str) -> Tuple[int, Optional[str]]:
        """Return engine evaluation and best move for the given FEN."""
        if fen in self._cache:
            return self._cache[fen]
        if self._engine is None:
            raise RuntimeError("Engine not initialized; use within context manager.")

        limit_kwargs = {"depth": self.depth}
        if self.nodes is not None and self.nodes > 0:
            limit_kwargs = {"nodes": self.nodes}

        board = chess.Board(fen)
        info = self._engine.analyse(board, chess.engine.Limit(**limit_kwargs))
        score = info["score"].white().score(mate_score=10000)
        if score is None:
            score = 0
        best: Optional[str] = None
        pv = info.get("pv")
        if pv:
            best = pv[0].uci()
        result = (int(score), best)
        self._cache[fen] = result
        return result

    def evaluate_fen(self, fen: str) -> int:
        """Compatibility helper returning only the evaluation component."""
        return self.analyse_fen(fen)[0]


__all__ = ["StockfishEngine", "resolve_engine_path", "StockfishUnavailableError"]
