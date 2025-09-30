from pathlib import Path

import pandas as pd

from cwa.data.pgn_parser import pgn_to_games_and_moves
from cwa.features.phases import annotate_phases


def load_sample_pgns() -> list[str]:
    sample_path = Path("tests/data/sample_small.pgn")
    raw = sample_path.read_text(encoding="utf-8").strip()
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


def test_phase_annotation_spans_all_phases():
    pgns = load_sample_pgns()
    games_df, moves_df = pgn_to_games_and_moves(pgns, username="sample")
    moves_df = annotate_phases(moves_df)
    phases = set(moves_df["phase"])
    assert "opening" in phases
    assert "middlegame" in phases
    assert "endgame" in phases
