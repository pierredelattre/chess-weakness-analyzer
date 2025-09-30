"""Convert Chess.com PGNs into structured DataFrames."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess
import chess.pgn
import pandas as pd

from cwa.utils import get_logger, slugify

LOGGER = get_logger("data.pgn_parser")


CLOCK_PATTERN = re.compile(r"\[%clk\s+([0-9:]+)\]")


def parse_clock_seconds(comment: str) -> Optional[float]:
    """Parse the clock annotation from a move comment."""
    match = CLOCK_PATTERN.search(comment)
    if not match:
        return None
    time_str = match.group(1)
    parts = [int(p) for p in time_str.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours = 0
        minutes = 0
        seconds = parts[0]
    return float(hours * 3600 + minutes * 60 + seconds)


def infer_time_class(headers: chess.pgn.Headers) -> str:
    """Attempt to infer Chess.com time class."""
    time_class = headers.get("TimeClass")
    if time_class:
        return time_class.lower()

    time_control = headers.get("TimeControl", "").lower()
    if not time_control or time_control in {"-", "*"}:
        return "unknown"
    if time_control == "unlimited" or "/" in time_control:
        return "daily"

    base = time_control.split("+")[0]
    try:
        base_seconds = int(base)
    except ValueError:
        return "unknown"

    if base_seconds <= 180:
        return "bullet"
    if base_seconds <= 600:
        return "blitz"
    if base_seconds <= 1800:
        return "rapid"
    return "daily"


def resolve_player_color(headers: chess.pgn.Headers, username: str) -> Tuple[str, str]:
    """Determine player's color and opponent name."""
    username_l = username.lower()
    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    if username_l == white:
        return "white", headers.get("Black", "Unknown")
    if username_l == black:
        return "black", headers.get("White", "Unknown")
    LOGGER.warning("Username %s not found in PGN headers; defaulting to white", username)
    return "white", headers.get("Black", "Unknown")


def make_game_id(headers: chess.pgn.Headers) -> str:
    """Construct a deterministic game identifier."""
    components = [
        headers.get("Site", "chess.com"),
        headers.get("UTCDate", ""),
        headers.get("UTCTime", ""),
        headers.get("Round", ""),
    ]
    raw = "_".join(filter(None, components))
    return slugify(raw)[:64]


def pgn_to_games_and_moves(pgns: Iterable[str], username: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse PGN strings and return games and moves DataFrames."""
    games_records: List[dict] = []
    moves_records: List[dict] = []

    for raw in pgns:
        if not raw.strip():
            continue
        game = chess.pgn.read_game(io.StringIO(raw))
        if game is None:
            continue
        headers = game.headers
        player_color, opponent = resolve_player_color(headers, username)
        game_id = make_game_id(headers)
        result = headers.get("Result", "*")
        game_url = headers.get("Site", "")
        time_control = headers.get("TimeControl", "")
        time_class = infer_time_class(headers)
        rated = headers.get("Rated", "")
        is_rated = rated.lower() in {"true", "yes", "1"}

        games_records.append(
            {
                "game_id": game_id,
                "site": headers.get("Site", ""),
                "url": headers.get("Link", game_url),
                "utc_date": headers.get("UTCDate", ""),
                "utc_time": headers.get("UTCTime", ""),
                "time_control": time_control,
                "time_class": time_class,
                "result": result,
                "color": player_color,
                "opponent": opponent,
                "eco": headers.get("ECO", ""),
                "opening": headers.get("Opening", ""),
                "termination": headers.get("Termination", ""),
                "rated": is_rated,
            }
        )

        board = game.board()
        ply = 0
        for node in game.mainline():
            move = node.move
            ply += 1
            side_to_move = "white" if board.turn == chess.WHITE else "black"
            fen_before = board.fen()
            san = board.san(move)
            uci = move.uci()
            board.push(move)
            fen_after = board.fen()
            comment = node.comment or ""
            clock_seconds = parse_clock_seconds(comment)
            moves_records.append(
                {
                    "game_id": game_id,
                    "ply": ply,
                    "move_number": (ply + 1) // 2,
                    "side_to_move": side_to_move,
                    "player_move": side_to_move == player_color,
                    "san": san,
                    "uci": uci,
                    "fen_before": fen_before,
                    "fen_after": fen_after,
                    "clock_seconds": clock_seconds,
                    "comment": comment,
                    "nags": list(sorted(node.nags)) if node.nags else [],
                }
            )
        # capture outcome for players without result header
        if result == "*":
            termination = headers.get("Termination", "")
            if "checkmate" in termination.lower():
                if board.turn == chess.WHITE:
                    result = "0-1"
                else:
                    result = "1-0"

    games_df = pd.DataFrame(games_records)
    moves_df = pd.DataFrame(moves_records)
    return games_df, moves_df


__all__ = ["pgn_to_games_and_moves"]
