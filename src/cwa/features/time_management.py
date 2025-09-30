"""Time management analytics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from cwa.utils import get_logger, pearsonr_safe

LOGGER = get_logger("features.time")


SCRAMBLE_THRESHOLD = 20  # seconds


def compute_time_features(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Compute time usage stats for player moves."""
    player_moves = moves_df[
        moves_df["player_move"]
        & moves_df["clock_seconds"].notna()
        & moves_df["abs_delta_cp"].notna()
    ].copy()

    if player_moves.empty:
        return pd.DataFrame(
            [
                {
                    "metric": "available",
                    "value": 0.0,
                    "detail": "0 = clock data missing in PGNs",
                }
            ]
        )

    player_moves.sort_values(["game_id", "ply"], inplace=True)
    player_moves["clock_seconds"] = player_moves["clock_seconds"].astype(float)
    player_moves["time_spent"] = player_moves.groupby("game_id")["clock_seconds"].shift(1) - player_moves["clock_seconds"]
    player_moves["time_spent"] = player_moves["time_spent"].clip(lower=0).fillna(0)

    avg_time = player_moves["time_spent"].mean()
    scramble_rate = (player_moves["clock_seconds"] <= SCRAMBLE_THRESHOLD).mean()
    blunders_in_scramble = player_moves[player_moves["clock_seconds"] <= SCRAMBLE_THRESHOLD][
        "is_blunder"
    ].mean()
    overall_blunder_rate = player_moves["is_blunder"].mean()

    correlation = pearsonr_safe(
        player_moves["clock_seconds"].tolist(), player_moves["abs_delta_cp"].tolist()
    )

    rows = [
        {"metric": "available", "value": 1.0, "detail": "1 = clock data present"},
        {"metric": "avg_time_spent", "value": float(avg_time), "detail": "seconds"},
        {
            "metric": "scramble_rate",
            "value": float(scramble_rate * 100),
            "detail": f"% moves under {SCRAMBLE_THRESHOLD}s",
        },
        {
            "metric": "scramble_blunder_rate",
            "value": float((blunders_in_scramble or 0) * 100),
            "detail": "Blunder % when under threshold",
        },
        {
            "metric": "overall_blunder_rate",
            "value": float((overall_blunder_rate or 0) * 100),
            "detail": "Blunders % overall",
        },
        {
            "metric": "time_blunder_correlation",
            "value": float(correlation) if correlation is not None else np.nan,
            "detail": "Pearson correlation (clock vs |delta|)",
        },
    ]
    return pd.DataFrame(rows)


__all__ = ["compute_time_features"]
