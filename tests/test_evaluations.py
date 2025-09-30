import pandas as pd

from cwa.features.evaluations import annotate_engine_evaluations


class DummyEngine:
    def __init__(self, mapping: dict[str, int]):
        self.mapping = mapping

    def analyse_fen(self, fen: str):
        return self.mapping[fen], None

    def evaluate_fen(self, fen: str) -> int:
        return self.mapping[fen]


def test_engine_annotations_compute_delta_and_blunder():
    data = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "ply": 1,
                "move_number": 1,
                "side_to_move": "white",
                "player_move": True,
                "san": "e4",
                "uci": "e2e4",
                "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            },
            {
                "game_id": "g1",
                "ply": 2,
                "move_number": 1,
                "side_to_move": "black",
                "player_move": True,
                "san": "e5",
                "uci": "e7e5",
                "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "fen_after": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            },
            {
                "game_id": "g1",
                "ply": 3,
                "move_number": 2,
                "side_to_move": "white",
                "player_move": True,
                "san": "Qg4",
                "uci": "d1g4",
                "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "fen_after": "rnbqkbnr/pppp1ppp/8/4p3/6Q1/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
            },
        ]
    )

    engine = DummyEngine(
        {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": 20,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": 50,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": 250,
            "rnbqkbnr/pppp1ppp/8/4p3/6Q1/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2": -200,
        }
    )

    annotated = annotate_engine_evaluations(
        moves_df=data,
        engine=engine,
        blunder_threshold=-200,
    )

    white_move = annotated.iloc[0]
    assert white_move["eval_before_cp"] == 20
    assert white_move["eval_after_cp"] == 50
    assert white_move["delta_cp"] == 30
    assert not bool(white_move["is_blunder"])

    black_move = annotated.iloc[1]
    assert black_move["eval_before_cp"] == -50
    assert black_move["eval_after_cp"] == -250
    assert black_move["delta_cp"] == -200
    assert bool(black_move["is_blunder"]) is True

    blunder_move = annotated.iloc[2]
    assert blunder_move["delta_cp"] == -450
    assert bool(blunder_move["is_blunder"]) is True
