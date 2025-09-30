"""Streamlit dashboard for Chess Weakness Analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cwa.analyzer import AnalysisArtifacts, run_analyze
from cwa.config import get_settings
from cwa.data.storage import processed_path, read_parquet
from cwa.reports.coaching import CoachingInsights
from cwa.utils import configure_logging


st.set_page_config(page_title="Chess Weakness Analyzer", layout="wide")
configure_logging()
settings = get_settings()


def load_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        return read_parquet(path)
    except FileNotFoundError:
        return pd.DataFrame()


def load_processed(username: str) -> dict[str, pd.DataFrame]:
    data = {}
    for name in [
        "player_summary",
        "openings_summary",
        "blunders",
        "time_stats",
        "non_conversion",
        "tactical_events",
        "games",
        "moves",
    ]:
        data[name] = load_parquet_safe(processed_path(settings, username, name))
    return data


def compute_kpis(player_summary: pd.DataFrame) -> dict[str, float]:
    if player_summary.empty:
        return {"games": 0, "score_pct": 0.0, "acpl": 0.0, "blunders": 0.0}
    overall = player_summary[player_summary["segment"] == "overall"].iloc[0]
    return {
        "games": overall["games"],
        "score_pct": overall["score_pct"],
        "acpl": overall["acpl"],
        "blunders": overall["blunders_per_100"],
    }


def filter_games(games: pd.DataFrame, time_classes: list[str], colors: list[str]) -> pd.DataFrame:
    filtered = games
    if time_classes:
        filtered = filtered[filtered["time_class"].isin(time_classes)]
    if colors:
        filtered = filtered[filtered["color"].isin(colors)]
    return filtered


def filter_moves(moves: pd.DataFrame, game_ids: pd.Series) -> pd.DataFrame:
    if moves.empty:
        return moves
    return moves[moves["game_id"].isin(game_ids)]


def render_coaching_panel(data: dict[str, pd.DataFrame]) -> None:
    try:
        insights = CoachingInsights.from_data(
            games_df=data["games"],
            player_moves=data["moves"][data["moves"]["player_move"]]
            if not data["moves"].empty
            else pd.DataFrame(),
            blunders=data["blunders"],
            openings=data["openings_summary"],
            non_conversion=data["non_conversion"],
            tactical=data["tactical_events"],
            time_stats=data["time_stats"],
        )
        st.markdown(insights.to_markdown())
    except Exception as exc:  # pragma: no cover - defensive for dashboard runtime
        st.warning(f"Impossible de générer les recommandations: {exc}")


def render_openings_chart(openings: pd.DataFrame) -> None:
    if openings.empty:
        st.info("Pas de données d'ouverture.")
        return
    top = openings.sort_values("avg_acpl", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["opening"], top["avg_acpl"], color="tomato")
    ax.invert_yaxis()
    ax.set_xlabel("ACPL")
    ax.set_title("Top ouvertures problématiques")
    st.pyplot(fig)


def render_trend_chart(moves: pd.DataFrame) -> None:
    if moves.empty:
        st.info("Pas de coups analysés.")
        return
    if "utc_date" not in moves.columns:
        st.info("Données temporelles manquantes.")
        return
    moves["utc_date"] = pd.to_datetime(moves["utc_date"])
    daily = (
        moves.groupby(moves["utc_date"].dt.date)["abs_delta_cp"].mean().reset_index(name="acpl")
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(daily["utc_date"], daily["acpl"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("ACPL")
    ax.set_title("Tendance ACPL")
    st.pyplot(fig)


def render_histogram(moves: pd.DataFrame) -> None:
    if moves.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(moves["delta_cp"].dropna(), bins=30, color="steelblue", alpha=0.7)
    ax.set_title("Distribution des erreurs (Δcp)")
    ax.set_xlabel("Δcp")
    st.pyplot(fig)


def render_blunders_table(blunders: pd.DataFrame) -> None:
    if blunders.empty:
        st.info("Aucune gaffe détectée.")
        return
    view = blunders[["game_id", "ply", "san", "phase", "delta_cp", "url"]].copy()
    view["lien"] = view["url"]
    st.dataframe(view.drop(columns=["url"]), use_container_width=True)


def render_openings_table(openings: pd.DataFrame) -> None:
    if openings.empty:
        return
    table = openings.sort_values("avg_acpl", ascending=False).head(10)
    st.dataframe(table, use_container_width=True)


def render_time_stats(time_stats: pd.DataFrame) -> None:
    if time_stats.empty:
        st.info("Stats de temps indisponibles.")
        return
    st.dataframe(time_stats, use_container_width=True)


def main() -> None:
    st.sidebar.header("Paramètres")
    username = st.sidebar.text_input("Username Chess.com", value=settings.chesscom_username or "")
    if not username:
        st.stop()

    time_class_filter = st.sidebar.multiselect(
        "Cadence", options=["bullet", "blitz", "rapid", "daily"]
    )
    color_filter = st.sidebar.multiselect("Couleur", options=["white", "black"])
    reanalyze = st.sidebar.button("(Re)Analyser")

    data = load_processed(username)

    if reanalyze:
        try:
            artifacts = run_analyze(username=username, settings=settings)
            data = load_processed(username)
            st.sidebar.success("Analyse mise à jour.")
        except Exception as exc:  # pragma: no cover - UI flow
            st.sidebar.error(f"Analyse impossible: {exc}")

    games_filtered = filter_games(data["games"], time_class_filter, color_filter)
    moves_filtered = filter_moves(data["moves"], games_filtered["game_id"]) if not games_filtered.empty else data["moves"]

    st.title("Chess Weakness Analyzer")
    kpis = compute_kpis(data["player_summary"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parties", f"{kpis['games']}")
    col2.metric("Score %", f"{kpis['score_pct']:.1f}")
    col3.metric("ACPL", f"{kpis['acpl']:.1f}")
    col4.metric("Blunders/100", f"{kpis['blunders']:.1f}")

    st.subheader("Ouvertures à risques")
    render_openings_chart(data["openings_summary"])

    st.subheader("Tendance ACPL")
    render_trend_chart(moves_filtered.merge(data["games"][['game_id', 'utc_date']], on='game_id', how='left'))

    st.subheader("Répartition des erreurs")
    render_histogram(moves_filtered)

    st.subheader("Principales gaffes")
    render_blunders_table(data["blunders"])

    st.subheader("Insights & recommandations")
    render_coaching_panel(data)

    st.subheader("Stats temps")
    render_time_stats(data["time_stats"])

    st.subheader("Ouvertures (table)")
    render_openings_table(data["openings_summary"])


if __name__ == "__main__":
    main()
