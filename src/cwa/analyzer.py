"""High-level orchestration for fetching and analyzing Chess.com games."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

from cwa.config import AppSettings, get_settings
from cwa.data.ingest_chesscom import FetchResult, download_archives, load_cached_archives, to_pgns
from cwa.data.storage import processed_path, read_parquet
from cwa.data.pgn_parser import pgn_to_games_and_moves
from cwa.engine.stockfish import StockfishEngine, resolve_engine_path
from cwa.features.aggregate import (
    build_openings_summary,
    persist_outputs,
    prepare_blunders_table,
)
from cwa.features.evaluations import (
    annotate_engine_evaluations,
    blunder_events,
    detect_non_conversion,
    player_move_summary,
)
from cwa.features.errors import detect_tactical_events
from cwa.features.openings import fill_missing_openings
from cwa.features.phases import annotate_phases
from cwa.features.time_management import compute_time_features
from cwa.reports.coaching import CoachingInsights
from cwa.utils import get_logger

LOGGER = get_logger("analyzer")


def _safe_username(username: str) -> str:
    return username.lower().replace(" ", "_")


def _analysis_meta_path(settings: AppSettings, username: str) -> Path:
    return settings.processed_dir() / f"{_safe_username(username)}_analysis_meta.json"


def _archive_key(archive: FetchResult) -> str:
    try:
        parts = archive.url.rstrip("/").split("/")
        year = int(parts[-2])
        month = int(parts[-1])
        return f"{year:04d}-{month:02d}"
    except Exception:  # pragma: no cover - defensive fallback
        stem = archive.path.stem
        for token in stem.split("_"):
            if len(token) == 4 and token.isdigit():
                year = int(token)
            if len(token) == 2 and token.isdigit():
                month = int(token)
        return stem


def _load_existing_games_moves(
    settings: AppSettings, username: str
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    games_path = processed_path(settings, username, "games")
    moves_path = processed_path(settings, username, "moves")
    if not games_path.exists() or not moves_path.exists():
        return None, None
    try:
        games_df = read_parquet(games_path)
        moves_df = read_parquet(moves_path)
    except Exception as exc:  # pragma: no cover - corrupted parquet
        LOGGER.warning("Failed to load cached games/moves for %s: %s", username, exc)
        return None, None
    return games_df, moves_df


def _load_analysis_metadata(settings: AppSettings, username: str) -> tuple[Set[str], Optional[int], Optional[int]]:
    meta_path = _analysis_meta_path(settings, username)
    if not meta_path.exists():
        return set(), None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        archives = set(str(item) for item in meta.get("archives", []))
        depth = meta.get("engine_depth")
        nodes = meta.get("engine_nodes")
        return archives, depth, nodes
    except Exception as exc:  # pragma: no cover - invalid metadata
        LOGGER.warning("Failed to read analysis metadata for %s: %s", username, exc)
        return set(), None, None


def _write_analysis_metadata(
    settings: AppSettings,
    username: str,
    archives: Set[str],
    engine_depth: Optional[int],
    engine_nodes: Optional[int],
) -> None:
    meta_path = _analysis_meta_path(settings, username)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "archives": sorted(archives),
                "engine_depth": engine_depth,
                "engine_nodes": engine_nodes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass
class AnalysisArtifacts:
    games: pd.DataFrame
    moves: pd.DataFrame
    player_moves: pd.DataFrame
    blunders: pd.DataFrame
    non_conversion: pd.DataFrame
    openings_summary: pd.DataFrame
    time_stats: pd.DataFrame
    tactical_events: pd.DataFrame
    coaching: CoachingInsights


def run_fetch(
    username: str,
    settings: Optional[AppSettings] = None,
    max_months: Optional[int] = None,
    force: bool = False,
) -> List[FetchResult]:
    """Fetch Chess.com archives for a username."""
    settings = settings or get_settings()
    LOGGER.info("Fetching archives for %s", username)
    return download_archives(username=username, settings=settings, max_months=max_months, force=force)


def run_analyze(
    username: str,
    settings: Optional[AppSettings] = None,
    engine_depth: Optional[int] = None,
    engine_nodes: Optional[int] = None,
    max_months: Optional[int] = None,
    pgn_files: Optional[Iterable[Path]] = None,
) -> AnalysisArtifacts:
    """Execute the full analysis pipeline for ``username``."""
    settings = settings or get_settings()
    pgns: List[str] = []
    current_depth = engine_depth if engine_depth is not None else settings.engine_depth
    current_nodes = engine_nodes if engine_nodes is not None else settings.engine_nodes

    if pgn_files:
        for path in pgn_files:
            pgns.extend(load_local_pgns(path))

    archives = load_cached_archives(username, settings)
    if max_months is not None and max_months > 0:
        archives = archives[-max_months:]
    archive_keys = [_archive_key(archive) for archive in archives]
    LOGGER.info(
        "Loaded %s cached archive batches for %s%s",
        len(archives),
        username,
        f" (limit={max_months} months)" if max_months else "",
    )
    if archives:
        for archive in archives:
            pgns.extend(to_pgns(archive.payload))

    if not pgns:
        raise RuntimeError("No games available. Fetch archives or provide PGN files.")
    LOGGER.info("Parsing %s PGNs for %s", len(pgns), username)

    games_df, moves_df = pgn_to_games_and_moves(pgns, username=username)
    if games_df.empty or moves_df.empty:
        raise RuntimeError("Failed to parse PGNs; games or moves dataframe empty.")
    LOGGER.info("Parsed %s games / %s moves", len(games_df), len(moves_df))

    # deduplicate by game_id just in case
    games_df = games_df.drop_duplicates(subset=["game_id"])
    moves_df = moves_df.drop_duplicates(subset=["game_id", "ply"])
    moves_df = moves_df.merge(
        games_df[["game_id", "utc_date"]], on="game_id", how="left"
    )
    requested_game_ids: Set[str] = set(games_df["game_id"].tolist())

    existing_games_df, existing_moves_df = _load_existing_games_moves(settings, username)
    existing_archives, cached_depth, cached_nodes = _load_analysis_metadata(settings, username)

    engine_override = engine_depth is not None or engine_nodes is not None
    engine_changed = False
    if cached_depth is not None and cached_depth != current_depth:
        engine_changed = True
    if cached_nodes is not None and cached_nodes != current_nodes:
        engine_changed = True
    if engine_override and (cached_depth != current_depth or cached_nodes != current_nodes):
        engine_changed = True
    if engine_changed and (existing_games_df is not None or existing_moves_df is not None):
        LOGGER.info(
            "Engine configuration change detected (depth %s→%s, nodes %s→%s); recomputing all games.",
            cached_depth,
            current_depth,
            cached_nodes,
            current_nodes,
        )
        existing_games_df = None
        existing_moves_df = None
        existing_archives = set()

    existing_game_ids: Set[str] = set(existing_games_df["game_id"].tolist()) if existing_games_df is not None else set()
    existing_requested_ids = requested_game_ids & existing_game_ids
    new_game_ids = requested_game_ids - existing_game_ids

    reuse_only = not new_game_ids and existing_games_df is not None and existing_moves_df is not None

    if reuse_only:
        LOGGER.info(
            "No new games detected for %s; reusing cached evaluations (games=%s).",
            username,
            len(existing_requested_ids),
        )
        combined_games_df = existing_games_df[existing_games_df["game_id"].isin(existing_requested_ids)].copy()
        combined_moves_df = existing_moves_df[existing_moves_df["game_id"].isin(existing_requested_ids)].copy()
        new_moves_df = pd.DataFrame(columns=moves_df.columns)
    else:
        existing_games_subset = (
            existing_games_df[existing_games_df["game_id"].isin(existing_requested_ids)].copy()
            if existing_games_df is not None
            else pd.DataFrame()
        )
        existing_moves_subset = (
            existing_moves_df[existing_moves_df["game_id"].isin(existing_requested_ids)].copy()
            if existing_moves_df is not None
            else pd.DataFrame()
        )

        new_games_df = games_df[games_df["game_id"].isin(new_game_ids)].copy()
        new_moves_df = moves_df[moves_df["game_id"].isin(new_game_ids)].copy()

        if new_game_ids:
            LOGGER.info(
                "Evaluating %s new games with Stockfish (existing cached: %s).",
                len(new_game_ids),
                len(existing_requested_ids),
            )
            engine_path = resolve_engine_path(settings)
            LOGGER.info("Using Stockfish binary at %s", engine_path)
            LOGGER.info("Evaluating moves with Stockfish (depth=%s, nodes=%s)", current_depth, current_nodes)

            with StockfishEngine(engine_path, depth=current_depth, nodes=current_nodes) as engine:
                new_moves_df = annotate_engine_evaluations(
                    new_moves_df,
                    engine,
                    blunder_threshold=settings.blunder_cp,
                )
            LOGGER.info("Engine annotations complete for new games")
        else:
            LOGGER.info("All requested games already cached; skipping Stockfish run.")

        if not existing_games_subset.empty:
            combined_games_df = pd.concat([existing_games_subset, new_games_df], ignore_index=True)
        else:
            combined_games_df = new_games_df

        if not existing_moves_subset.empty:
            combined_moves_df = pd.concat([existing_moves_subset, new_moves_df], ignore_index=True)
        else:
            combined_moves_df = new_moves_df

    if combined_games_df.empty or combined_moves_df.empty:
        raise RuntimeError("No moves available after caching logic; analysis cannot continue.")

    combined_games_df = combined_games_df.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    combined_moves_df = (
        combined_moves_df.drop_duplicates(subset=["game_id", "ply"])
        .sort_values(["game_id", "ply"])
        .reset_index(drop=True)
    )

    LOGGER.info("Annotating phases for %s moves", len(combined_moves_df))
    combined_moves_df = annotate_phases(combined_moves_df)

    games_df = fill_missing_openings(combined_games_df, combined_moves_df)
    player_moves_df = player_move_summary(combined_moves_df)
    non_conversion_df = detect_non_conversion(
        combined_moves_df,
        win_threshold=settings.win_thresh_cp,
        window=settings.non_conversion_window,
    )
    blunders_df = blunder_events(combined_moves_df)
    tactical_df = detect_tactical_events(combined_moves_df)
    blunders_detail_df = prepare_blunders_table(blunders_df, games_df, combined_moves_df)
    LOGGER.info(
        "Derived player metrics: %s player moves, %s blunders, %s tactical events",
        len(player_moves_df),
        len(blunders_detail_df),
        len(tactical_df),
    )

    openings_summary_df = build_openings_summary(games_df, player_moves_df)
    time_stats_df = compute_time_features(combined_moves_df)
    LOGGER.info(
        "Opening enrichment complete: %s openings tracked (games=%s)",
        len(openings_summary_df),
        openings_summary_df["games"].sum() if not openings_summary_df.empty else 0,
    )

    coaching = CoachingInsights.from_data(
        games_df=games_df,
        player_moves=player_moves_df,
        blunders=blunders_detail_df,
        openings=openings_summary_df,
        non_conversion=non_conversion_df,
        tactical=tactical_df,
        time_stats=time_stats_df,
    )

    persist_outputs(
        username=username,
        settings=settings,
        games_df=games_df,
        player_moves_df=player_moves_df,
        full_moves_df=combined_moves_df,
        blunders_df=blunders_detail_df,
        non_conversion_df=non_conversion_df,
        openings_summary_df=openings_summary_df,
        time_stats_df=time_stats_df,
        tactical_df=tactical_df,
    )
    LOGGER.info("Artifacts persisted for %s", username)

    updated_archives = existing_archives.union(archive_keys)
    _write_analysis_metadata(settings, username, updated_archives, current_depth, current_nodes)

    return AnalysisArtifacts(
        games=games_df,
        moves=combined_moves_df,
        player_moves=player_moves_df,
        blunders=blunders_df,
        non_conversion=non_conversion_df,
        openings_summary=openings_summary_df,
        time_stats=time_stats_df,
        tactical_events=tactical_df,
        coaching=coaching,
    )


def load_local_pgns(path: Path) -> List[str]:
    """Split a PGN file that may contain multiple games into individual strings."""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    chunks = raw.split("\n\n[Event ")
    pgns = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            pgns.append(chunk)
        else:
            pgns.append("[Event " + chunk)
    return pgns


__all__ = ["run_fetch", "run_analyze", "AnalysisArtifacts"]
