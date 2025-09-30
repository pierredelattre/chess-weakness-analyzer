"""Aggregation of analysis outputs and persistence to Parquet."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from cwa.config import AppSettings
from cwa.data.storage import append_parquet, processed_path, write_parquet
from cwa.features.evaluations import blunder_events
from cwa.features.openings import aggregate_opening_performance
from cwa.features.time_management import compute_time_features
from cwa.utils import chess_result_to_score


def build_player_summary(games_df: pd.DataFrame, player_moves_df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate player summary tables."""
    if games_df.empty:
        return pd.DataFrame()

    games_df = games_df.copy()
    games_df["score"] = games_df.apply(
        lambda r: chess_result_to_score(r["result"], r["color"]), axis=1
    )

    rows = []
    total_games = len(games_df)
    if total_games:
        rows.append(
            {
                "segment": "overall",
                "key": "all",
                "games": total_games,
                "score_pct": games_df["score"].mean() * 100,
                "acpl": player_moves_df["abs_delta_cp"].mean(),
                "blunders_per_100": 100
                * player_moves_df["is_blunder"].sum()
                / max(1, len(player_moves_df)),
            }
        )

    # by color
    for color, group in games_df.groupby("color"):
        rows.append(
            {
                "segment": "color",
                "key": color,
                "games": len(group),
                "score_pct": group["score"].mean() * 100 if len(group) else 0,
                "acpl": player_moves_df[player_moves_df["side_to_move"] == color][
                    "abs_delta_cp"
                ].mean(),
                "blunders_per_100": 100
                * player_moves_df[
                    (player_moves_df["side_to_move"] == color) & player_moves_df["is_blunder"]
                ].shape[0]
                / max(
                    1,
                    player_moves_df[player_moves_df["side_to_move"] == color].shape[0],
                ),
            }
        )

    # by time_class
    for time_class, group in games_df.groupby("time_class"):
        rows.append(
            {
                "segment": "time_class",
                "key": time_class,
                "games": len(group),
                "score_pct": group["score"].mean() * 100 if len(group) else 0,
                "acpl": player_moves_df[player_moves_df["game_id"].isin(group["game_id"])][
                    "abs_delta_cp"
                ].mean(),
                "blunders_per_100": 100
                * player_moves_df[
                    (player_moves_df["game_id"].isin(group["game_id"]))
                    & player_moves_df["is_blunder"]
                ].shape[0]
                / max(
                    1,
                    player_moves_df["game_id"].isin(group["game_id"]).sum(),
                ),
            }
        )

    # by phase
    for phase, group in player_moves_df.groupby("phase"):
        rows.append(
            {
                "segment": "phase",
                "key": phase,
                "games": games_df[games_df["game_id"].isin(group["game_id"])][
                    "game_id"
                ].nunique(),
                "score_pct": None,
                "acpl": group["abs_delta_cp"].mean(),
                "blunders_per_100": 100
                * group[group["is_blunder"]].shape[0]
                / max(1, len(group)),
            }
        )

    return pd.DataFrame(rows)


def persist_outputs(
    username: str,
    settings: AppSettings,
    games_df: pd.DataFrame,
    player_moves_df: pd.DataFrame,
    full_moves_df: pd.DataFrame,
    blunders_df: pd.DataFrame,
    non_conversion_df: pd.DataFrame,
    openings_summary_df: pd.DataFrame,
    time_stats_df: pd.DataFrame,
    tactical_df: pd.DataFrame,
) -> None:
    """Persist core outputs to Parquet files."""
    write_parquet(
        build_player_summary(games_df, player_moves_df),
        processed_path(settings, username, "player_summary"),
    )
    write_parquet(openings_summary_df, processed_path(settings, username, "openings_summary"))
    write_parquet(blunders_df, processed_path(settings, username, "blunders"))
    write_parquet(time_stats_df, processed_path(settings, username, "time_stats"))
    write_parquet(non_conversion_df, processed_path(settings, username, "non_conversion"))
    write_parquet(tactical_df, processed_path(settings, username, "tactical_events"))
    append_parquet(
        full_moves_df,
        processed_path(settings, username, "moves"),
        dedupe_keys=["game_id", "ply"],
    )
    append_parquet(
        games_df,
        processed_path(settings, username, "games"),
        dedupe_keys=["game_id"],
    )


def build_openings_summary(
    games_df: pd.DataFrame,
    player_moves_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience wrapper for opening aggregation."""
    return aggregate_opening_performance(games_df, player_moves_df, player_moves_df)


def prepare_blunders_table(
    blunders_df: pd.DataFrame,
    games_df: pd.DataFrame,
    moves_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enhance blunders with game URLs and SAN context."""
    games_meta = games_df[["game_id", "url", "utc_date", "time_class", "opening"]]
    enriched = blunders_df.merge(games_meta, on="game_id", how="left")
    moves_context = moves_df[["game_id", "ply", "uci", "comment"]]
    enriched = enriched.merge(moves_context, on=["game_id", "ply"], how="left")
    return enriched


__all__ = [
    "build_player_summary",
    "persist_outputs",
    "build_openings_summary",
    "prepare_blunders_table",
]
