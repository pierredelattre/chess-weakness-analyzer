"""Tactical error heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chess
import pandas as pd

from cwa.utils import get_logger

LOGGER = get_logger("features.errors")

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass
class TacticalEvent:
    game_id: str
    ply: int
    move_number: int
    san: str
    issue: str
    detail: str
    severity: int


def detect_tactical_events(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Detect hanging pieces, SEE negatives, and simple fork/pin motifs."""
    events: List[dict] = []

    player_moves = moves_df[moves_df["player_move"]]
    for _, row in player_moves.iterrows():
        fen_before = row["fen_before"]
        fen_after = row["fen_after"]
        if not fen_before or not fen_after:
            continue
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        move = chess.Move.from_uci(row["uci"])

        player_color = chess.WHITE if row["side_to_move"] == "white" else chess.BLACK
        opponent_color = not player_color

        see_value = evaluate_exchange(board_before, move)
        if see_value is not None and see_value < 0:
            events.append(
                {
                    "game_id": row["game_id"],
                    "ply": row["ply"],
                    "move_number": row["move_number"],
                    "san": row["san"],
                    "issue": "negative_see",
                    "detail": f"SEE = {see_value}",
                    "severity": abs(see_value),
                }
            )

        hanging = detect_hanging_piece(board_after, player_color, opponent_color)
        if hanging:
            piece_symbol, square = hanging
            events.append(
                {
                    "game_id": row["game_id"],
                    "ply": row["ply"],
                    "move_number": row["move_number"],
                    "san": row["san"],
                    "issue": "hanging_piece",
                    "detail": f"{piece_symbol} hanging on {chess.square_name(square)}",
                    "severity": piece_value(piece_symbol),
                }
            )

        fork = detect_fork_against(board_after, player_color, opponent_color)
        if fork:
            attacker, targets = fork
            events.append(
                {
                    "game_id": row["game_id"],
                    "ply": row["ply"],
                    "move_number": row["move_number"],
                    "san": row["san"],
                    "issue": "fork_threat",
                    "detail": f"{attacker} attacking {', '.join(targets)}",
                    "severity": len(targets) * 100,
                }
            )

        pin = detect_pin(board_after, player_color, opponent_color)
        if pin:
            pinned_piece, direction = pin
            events.append(
                {
                    "game_id": row["game_id"],
                    "ply": row["ply"],
                    "move_number": row["move_number"],
                    "san": row["san"],
                    "issue": "pin",
                    "detail": f"{pinned_piece} pinned {direction}",
                    "severity": 150,
                }
            )

    return pd.DataFrame(events)


def detect_hanging_piece(board: chess.Board, player_color: chess.Color, opponent_color: chess.Color):
    """Return (piece_symbol, square) if a player's piece is hanging."""
    for square, piece in board.piece_map().items():
        if piece.color != player_color:
            continue
        if piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(player_color, square):
            return piece.symbol(), square
    return None


def detect_fork_against(board: chess.Board, player_color: chess.Color, opponent_color: chess.Color):
    """Return attacker symbol and list of targets if opponent threatens a fork."""
    targets: List[str] = []
    attacker_symbol = ""
    for square, piece in board.piece_map().items():
        if piece.color != opponent_color:
            continue
        attacked = [
            chess.square_name(sq)
            for sq in board.attacks(square)
            if (target_piece := board.piece_at(sq)) and target_piece.color == player_color and target_piece.piece_type != chess.KING
        ]
        high_value = [sq for sq in attacked if piece_value(board.piece_at(chess.parse_square(sq)).symbol()) >= 300]
        if len(high_value) >= 2:
            attacker_symbol = piece.symbol()
            targets = high_value
            break
    if attacker_symbol and targets:
        return attacker_symbol, targets
    return None


def detect_pin(board: chess.Board, player_color: chess.Color, opponent_color: chess.Color):
    """Detect if any player's piece is pinned to the king."""
    king_square = board.king(player_color)
    if king_square is None:
        return None
    for square, piece in board.piece_map().items():
        if piece.color != player_color or piece.piece_type == chess.KING:
            continue
        if board.is_pinned(player_color, square):
            direction = direction_to_square(square, king_square)
            return piece.symbol(), direction
    return None


def direction_to_square(src: int, dst: int) -> str:
    """Return algebraic direction from src to dst."""
    file_diff = chess.square_file(dst) - chess.square_file(src)
    rank_diff = chess.square_rank(dst) - chess.square_rank(src)
    return f"{file_diff:+}/{rank_diff:+}"


def piece_value(symbol: str) -> int:
    """Return simple piece value for a piece symbol."""
    piece = chess.Piece.from_symbol(symbol)
    return PIECE_VALUES.get(piece.piece_type, 0)


def evaluate_exchange(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Return a static exchange evaluation if supported by python-chess."""
    if hasattr(board, "static_exchange_evaluation"):
        try:
            return int(board.static_exchange_evaluation(move))
        except ValueError:
            return None
    if hasattr(board, "see"):
        try:
            return int(board.see(move))
        except ValueError:
            return None
    return None


__all__ = ["detect_tactical_events"]
