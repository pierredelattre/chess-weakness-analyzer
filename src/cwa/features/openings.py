"""Opening classification and aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from cwa.utils import chess_result_to_score

ECO_FALLBACKS: Dict[Tuple[str, ...], Tuple[str, str]] = {
    ("e4", "e5", "Nf3", "Nc6"): ("C50", "Italian Game"),
    ("e4", "e5", "Nf3", "Nc6", "Bb5"): ("C60", "Ruy Lopez"),
    ("e4", "c5"): ("B20", "Sicilian Defence"),
    ("e4", "c6"): ("B10", "Caro-Kann Defence"),
    ("e4", "e6"): ("C00", "French Defence"),
    ("d4", "d5", "c4"): ("D06", "Queen's Gambit"),
    ("d4", "Nf6", "c4", "g6"): ("E60", "King's Indian Defence"),
    ("c4",): ("A10", "English Opening"),
    ("Nf3", "d5", "g3"): ("A10", "Reti Opening"),
    ("d4", "d5", "Nf3", "Nf6", "c4"): ("D30", "Queen's Gambit Declined"),
    ("e4", "d5"): ("B01", "Scandinavian Defence"),
}


def fill_missing_openings(games_df: pd.DataFrame, moves_df: pd.DataFrame) -> pd.DataFrame:
    """Fill ECO/opening fields using fallback patterns."""
    games_df = games_df.copy()
    moves_by_game = moves_df.groupby("game_id")["san"].apply(list)

    for idx, row in games_df.iterrows():
        eco = row.get("eco") or ""
        opening = row.get("opening") or ""
        if eco and opening:
            continue
        moves = moves_by_game.get(row["game_id"], [])
        fallback = infer_opening(moves)
        if fallback:
            games_df.at[idx, "eco"] = fallback[0]
            games_df.at[idx, "opening"] = fallback[1]
        else:
            games_df.at[idx, "eco"] = eco or "UNK"
            games_df.at[idx, "opening"] = opening or "Unknown"
    return games_df


def infer_opening(moves: Sequence[str]) -> Tuple[str, str] | None:
    """Infer ECO/opening from the initial move sequence."""
    if not moves:
        return None
    for pattern, result in ECO_FALLBACKS.items():
        if len(moves) < len(pattern):
            continue
        if tuple(moves[: len(pattern)]) == pattern:
            return result
    return None


def aggregate_opening_performance(
    games_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate opening results by ECO, color, and time class."""
    # metrics_df is player moves with acpl/delta
    blunders_by_game = metrics_df.groupby("game_id")["is_blunder"].sum().rename("blunders")
    avg_acpl = metrics_df.groupby("game_id")["abs_delta_cp"].mean().rename("acpl")

    enriched = games_df.merge(blunders_by_game, on="game_id", how="left")
    enriched = enriched.merge(avg_acpl, on="game_id", how="left")

    enriched["blunders"] = enriched["blunders"].fillna(0)
    enriched["acpl"] = enriched["acpl"].fillna(0)

    enriched["score"] = enriched.apply(
        lambda r: chess_result_to_score(r["result"], r["color"]), axis=1
    )

    group_cols = ["eco", "opening", "color", "time_class"]
    aggregated = (
        enriched.groupby(group_cols)
        .agg(
            games=("game_id", "count"),
            score_pct=("score", lambda s: 100 * s.mean() if len(s) else 0),
            avg_acpl=("acpl", "mean"),
            blunders_per_game=("blunders", lambda s: s.sum() / len(s) if len(s) else 0),
        )
        .reset_index()
    )
    return aggregated


__all__ = ["fill_missing_openings", "aggregate_opening_performance", "infer_opening"]
