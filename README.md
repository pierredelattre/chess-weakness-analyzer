# Chess Weakness Analyzer (CWA)

Chess Weakness Analyzer ingests your Chess.com games, evaluates your play with Stockfish, and surfaces actionable weaknesses through a CLI workflow and a Streamlit dashboard.

## Features
- Batch ingestion of Chess.com archives with local caching and retry/backoff logic.
- PGN parsing into structured pandas DataFrames for games and moves.
- Stockfish-assisted analysis of your moves only (configurable depth/nodes) with adaptive sampling.
- Feature extraction for ACPL by phase, blunders, non-conversion, opening performance, tactical motifs, and time management.
- Aggregated analytics saved to Parquet, Markdown/HTML coaching reports, and interactive dashboards.
- Offline-first: once archives are cached and Stockfish resolved, all analysis runs locally.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # update values as needed
```

### CLI pipeline
```bash
# replace USERNAME with your Chess.com handle
python -m cwa.cli all --user USERNAME --max-months 6 --engine-depth 14

# Offline test using bundled PGN sample (no network required)
python -m cwa.cli analyze --user sample --pgn-file tests/data/sample_small.pgn --engine-depth 12
```

### Streamlit dashboard
```bash
streamlit run src/cwa/app_streamlit.py
```

## Configuration
Environment variables are managed via Pydantic settings (`src/cwa/config.py`). Key variables:
- `CHESSCOM_USERNAME`: default username for CLI/dashboards.
- `DATA_DIR`: folder for raw/processed data (`./data` by default).
- `STOCKFISH_PATH`: optional explicit path to Stockfish binary.
- `MAX_MONTHS`, `ENGINE_DEPTH`, `ENGINE_NODES`, `BLUNDER_CP`, `WIN_THRESH_CP`, `NON_CONVERSION_WINDOW`: analysis tunables.

## Stockfish
If `STOCKFISH_PATH` is not set, the analyzer attempts to download the latest Stockfish release matching your OS/architecture. If auto-download fails, you will receive actionable instructions.

## Development
```bash
make venv
make install
make test
```

### Sample data
A minimal PGN lives in `tests/data/sample_small.pgn` for offline testing.

## License
MIT
