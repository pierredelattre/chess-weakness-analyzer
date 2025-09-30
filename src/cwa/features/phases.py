"""Game phase classification based on material balance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import chess
import pandas as pd

from cwa.utils import get_logger

LOGGER = get_logger("features.phases")

PIECE_PHASE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 4 * PIECE_PHASE[chess.KNIGHT] + 4 * PIECE_PHASE[chess.BISHOP] + 4 * PIECE_PHASE[chess.ROOK] + 2 * PIECE_PHASE[chess.QUEEN]


def board_phase(board: chess.Board) -> str:
    """Return phase label (opening/middlegame/endgame) for a board."""
    phase_score = TOTAL_PHASE
    for piece_type, weight in PIECE_PHASE.items():
        phase_score -= weight * len(board.pieces(piece_type, chess.WHITE))
        phase_score -= weight * len(board.pieces(piece_type, chess.BLACK))

    if board.fullmove_number <= 8:
        return "opening"
    # clamp score
    if phase_score <= TOTAL_PHASE * 0.3:
        return "opening"
    if phase_score >= TOTAL_PHASE * 0.8:
        return "endgame"
    return "middlegame"


def annotate_phases(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``phase`` column to ``moves_df`` using ``fen_before``."""
    if moves_df.empty:
        moves_df["phase"] = []
        return moves_df

    phases: List[str] = []
    for fen in moves_df["fen_before"].fillna(""):
        if not fen:
            phases.append("unknown")
            continue
        board = chess.Board(fen)
        phases.append(board_phase(board))
    moves_df = moves_df.copy()
    moves_df["phase"] = phases
    return moves_df


__all__ = ["annotate_phases", "board_phase"]
