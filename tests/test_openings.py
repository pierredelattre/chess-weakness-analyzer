import pandas as pd

from cwa.features.openings import fill_missing_openings, infer_opening


def test_infer_opening_matches_known_pattern():
    eco, name = infer_opening(["e4", "c5"])
    assert eco == "B20"
    assert "Sicilian" in name


def test_infer_opening_uses_prefix_map():
    eco, name = infer_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"])
    assert eco.startswith("C")
    assert "Ruy" in name or "Espagnole" in name


def test_fill_missing_openings_uses_fallback():
    games_df = pd.DataFrame(
        [
            {"game_id": "g1", "eco": "", "opening": "", "time_class": "blitz", "color": "white"},
            {"game_id": "g2", "eco": "C50", "opening": "Italian Game", "time_class": "rapid", "color": "black"},
        ]
    )
    moves_df = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g1"],
            "san": ["e4", "e5", "Nf3", "Nc6"],
        }
    )
    filled = fill_missing_openings(games_df, moves_df)
    row = filled[filled["game_id"] == "g1"].iloc[0]
    assert row["eco"].startswith("C")
    assert "Italian" in row["opening"] or "Ital" in row["opening"]
