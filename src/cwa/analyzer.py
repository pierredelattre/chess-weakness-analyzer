"""High-level orchestration for fetching and analyzing Chess.com games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from cwa.config import AppSettings, get_settings
from cwa.data.ingest_chesscom import FetchResult, download_archives, load_cached_archives, to_pgns
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

    if pgn_files:
        for path in pgn_files:
            pgns.extend(load_local_pgns(path))

    archives = load_cached_archives(username, settings)
    if max_months is not None and max_months > 0:
        archives = archives[-max_months:]
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

    LOGGER.info("Annotating phases for %s moves", len(moves_df))

    moves_df = annotate_phases(moves_df)

    engine_path = resolve_engine_path(settings)
    LOGGER.info("Using Stockfish binary at %s", engine_path)
    depth = engine_depth if engine_depth is not None else settings.engine_depth
    nodes = engine_nodes if engine_nodes is not None else settings.engine_nodes
    LOGGER.info("Evaluating moves with Stockfish (depth=%s, nodes=%s)", depth, nodes)

    with StockfishEngine(engine_path, depth=depth, nodes=nodes) as engine:
        moves_df = annotate_engine_evaluations(
            moves_df,
            engine,
            blunder_threshold=settings.blunder_cp,
        )
    LOGGER.info("Engine annotations complete")

    player_moves_df = player_move_summary(moves_df)
    non_conversion_df = detect_non_conversion(
        moves_df,
        win_threshold=settings.win_thresh_cp,
        window=settings.non_conversion_window,
    )
    blunders_df = blunder_events(moves_df)
    tactical_df = detect_tactical_events(moves_df)
    blunders_detail_df = prepare_blunders_table(blunders_df, games_df, moves_df)
    LOGGER.info(
        "Derived player metrics: %s player moves, %s blunders, %s tactical events",
        len(player_moves_df),
        len(blunders_detail_df),
        len(tactical_df),
    )

    games_df = fill_missing_openings(games_df, moves_df)
    openings_summary_df = build_openings_summary(games_df, player_moves_df)
    time_stats_df = compute_time_features(moves_df)
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
        full_moves_df=moves_df,
        blunders_df=blunders_detail_df,
        non_conversion_df=non_conversion_df,
        openings_summary_df=openings_summary_df,
        time_stats_df=time_stats_df,
        tactical_df=tactical_df,
    )
    LOGGER.info("Artifacts persisted for %s", username)

    return AnalysisArtifacts(
        games=games_df,
        moves=moves_df,
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
