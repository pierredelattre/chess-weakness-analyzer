"""Engine-backed evaluation metrics for player moves."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from cwa.config import AppSettings
from cwa.engine.stockfish import StockfishEngine
from cwa.utils import get_logger

LOGGER = get_logger("features.evaluations")


def annotate_engine_evaluations(
    moves_df: pd.DataFrame,
    engine: StockfishEngine,
    blunder_threshold: int,
    skip_delta_threshold: int = 15,
) -> pd.DataFrame:
    """Annotate player's moves with engine evaluations.

    Adds the following columns for player moves:
    - eval_before_cp / eval_after_cp (player perspective)
    - eval_before_white / eval_after_white (raw engine score from White side)
    - delta_cp and abs_delta_cp
    - is_blunder flag when delta < blunder_threshold
    """

    if moves_df.empty:
        for column in [
            "eval_before_cp",
            "eval_after_cp",
            "eval_before_white",
            "eval_after_white",
            "delta_cp",
            "abs_delta_cp",
            "is_blunder",
            "low_impact",
        ]:
            moves_df[column] = []
        return moves_df

    moves_df = moves_df.copy()
    moves_df[
        [
            "eval_before_cp",
            "eval_after_cp",
            "eval_before_white",
            "eval_after_white",
            "delta_cp",
            "abs_delta_cp",
        ]
    ] = np.nan
    moves_df["is_blunder"] = False
    moves_df["low_impact"] = False
    moves_df["best_uci"] = None

    for idx, row in moves_df[moves_df["player_move"]].iterrows():
        fen_before = row["fen_before"]
        fen_after = row["fen_after"]
        if not fen_before or not fen_after:
            continue

        eval_before_white, best_uci = engine.analyse_fen(fen_before)
        eval_after_white = engine.evaluate_fen(fen_after)

        if row["side_to_move"] == "white":
            eval_before_player = eval_before_white
            eval_after_player = eval_after_white
        else:
            eval_before_player = -eval_before_white
            eval_after_player = -eval_after_white

        delta = eval_after_player - eval_before_player
        low_impact = abs(delta) < skip_delta_threshold

        moves_df.at[idx, "eval_before_cp"] = eval_before_player
        moves_df.at[idx, "eval_after_cp"] = eval_after_player
        moves_df.at[idx, "eval_before_white"] = eval_before_white
        moves_df.at[idx, "eval_after_white"] = eval_after_white
        moves_df.at[idx, "delta_cp"] = delta
        moves_df.at[idx, "abs_delta_cp"] = abs(delta)
        moves_df.at[idx, "is_blunder"] = bool(delta <= blunder_threshold)
        moves_df.at[idx, "low_impact"] = low_impact
        moves_df.at[idx, "best_uci"] = best_uci

    return moves_df


def acpl_by_group(moves_df: pd.DataFrame, group_fields: list[str]) -> pd.DataFrame:
    """Compute average centipawn loss for the player's moves by group."""
    player_moves = moves_df[moves_df["player_move"] & moves_df["abs_delta_cp"].notna()]
    if player_moves.empty:
        return pd.DataFrame(columns=group_fields + ["acpl"])
    grouped = player_moves.groupby(group_fields)["abs_delta_cp"].mean().reset_index()
    grouped = grouped.rename(columns={"abs_delta_cp": "acpl"})
    return grouped


def player_move_summary(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Return essential columns for player's moves only."""
    columns = [
        "game_id",
        "ply",
        "move_number",
        "side_to_move",
        "san",
        "uci",
        "phase",
        "fen_before",
        "eval_before_cp",
        "eval_after_cp",
        "delta_cp",
        "abs_delta_cp",
        "is_blunder",
        "low_impact",
        "best_uci",
    ]
    available = [c for c in columns if c in moves_df.columns]
    result = moves_df[moves_df["player_move"]][available].copy()
    if "uci" in result.columns and "played_uci" not in result.columns:
        result = result.rename(columns={"uci": "played_uci"})
    return result


def detect_non_conversion(
    moves_df: pd.DataFrame,
    win_threshold: int,
    window: int,
) -> pd.DataFrame:
    """Detect non-conversion events where winning eval drops to draw/loss within window."""
    player_moves = moves_df[moves_df["player_move"] & moves_df["eval_before_cp"].notna()]
    events: list[dict] = []

    for game_id, group in player_moves.groupby("game_id"):
        group = group.sort_values("ply")
        for idx, row in group.iterrows():
            if row["eval_before_cp"] < win_threshold:
                continue
            window_slice = group[group["ply"] > row["ply"]].head(window)
            if window_slice.empty:
                continue
            if any(window_slice["eval_after_cp"] < 0):
                events.append(
                    {
                        "game_id": game_id,
                        "ply": row["ply"],
                        "san": row["san"],
                        "eval_before_cp": row["eval_before_cp"],
                        "drop_to_cp": float(window_slice["eval_after_cp"].min()),
                        "window": len(window_slice),
                    }
                )
    return pd.DataFrame(events)


def blunder_events(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows corresponding to blunders."""
    columns = [
        "game_id",
        "ply",
        "move_number",
        "san",
        "phase",
        "eval_before_cp",
        "eval_after_cp",
        "delta_cp",
        "uci",
        "fen_before",
        "best_uci",
    ]
    player_moves = moves_df[moves_df["player_move"] & moves_df["is_blunder"]]
    available = [c for c in columns if c in player_moves.columns]
    blunders = player_moves[available].copy()
    if "uci" in blunders.columns and "played_uci" not in blunders.columns:
        blunders = blunders.rename(columns={"uci": "played_uci"})
    return blunders


__all__ = [
    "annotate_engine_evaluations",
    "acpl_by_group",
    "player_move_summary",
    "detect_non_conversion",
    "blunder_events",
]
