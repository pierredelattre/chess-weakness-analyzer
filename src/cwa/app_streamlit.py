"""Streamlit dashboard for Chess Weakness Analyzer."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import altair as alt
import random
import chess
import pandas as pd
import streamlit as st

from cwa.analyzer import run_analyze
from cwa.config import get_settings
from cwa.data.storage import processed_path, read_parquet
from cwa.engine.stockfish import StockfishEngine, resolve_engine_path
from cwa.reports.board_viz import render_position
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
    data: dict[str, pd.DataFrame] = {}
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
    if not data["moves"].empty:
        if "played_uci" not in data["moves"].columns and "uci" in data["moves"].columns:
            data["moves"] = data["moves"].rename(columns={"uci": "played_uci"})
        if "fen_before" not in data["moves"].columns:
            data["moves"]["fen_before"] = None
    if not data["blunders"].empty:
        if "played_uci" not in data["blunders"].columns and "uci" in data["blunders"].columns:
            data["blunders"] = data["blunders"].rename(columns={"uci": "played_uci"})
        if "fen_before" not in data["blunders"].columns:
            data["blunders"]["fen_before"] = None
    return data


def sanitize_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["utc_date"] = pd.to_datetime(result["utc_date"], errors="coerce")
    return result


def normalize_opening_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "opening" not in df.columns:
        return df
    result = df.copy()
    result["opening"] = result["opening"].fillna("")
    mask = result["opening"].isin(["", "UNK", "Unknown"])
    result.loc[mask, "opening"] = "Non identifiée"
    return result


def compute_kpis(player_summary: pd.DataFrame) -> dict[str, float]:
    if player_summary.empty:
        return {"games": 0.0, "score_pct": 0.0, "acpl": 0.0, "blunders": 0.0}
    overall = player_summary[player_summary["segment"] == "overall"].iloc[0]
    return {
        "games": float(overall["games"]),
        "score_pct": float(overall["score_pct"]),
        "acpl": float(overall["acpl"]),
        "blunders": float(overall["blunders_per_100"]),
    }


def badge_from_threshold(metric: str, value: float) -> str:
    if metric == "acpl":
        return "🟢 **OK**" if value <= 150 else "🟠 À travailler"
    if metric == "blunders":
        return "🟢 **OK**" if value <= 5 else "🟠 À travailler"
    if metric == "score_pct":
        return "🟢 **OK**" if value >= 50 else "🟠 À travailler"
    return ""


def build_training_queue(
    blunders: pd.DataFrame,
    openings_summary: pd.DataFrame,
    top_n: int,
    per_opening: int,
) -> list[dict]:
    if blunders.empty:
        return []
    focus = openings_summary.sort_values("avg_acpl", ascending=False)
    focus = focus[focus["games"] > 0]
    focus_openings = focus["opening"].head(top_n).tolist()
    queue: list[dict] = []
    seen = set()
    for opening in focus_openings:
        subset = blunders[blunders["opening"] == opening].copy()
        if subset.empty:
            continue
        subset["weight"] = subset["delta_cp"].abs()
        subset = subset.sort_values("weight", ascending=False)
        subset = subset.drop_duplicates(["game_id", "ply"])
        for _, row in subset.head(per_opening).iterrows():
            key = (row["game_id"], int(row["ply"]))
            if key in seen:
                continue
            seen.add(key)
            queue.append(row.to_dict())
    random.shuffle(queue)
    return queue


def ensure_best_move(exercise: dict, app_settings) -> tuple[Optional[str], Optional[str]]:
    fen = exercise.get("fen_before")
    if not fen:
        return exercise.get("best_uci"), None

    cache = st.session_state.setdefault("best_move_cache", {})
    if exercise.get("best_uci"):
        move_uci = exercise["best_uci"]
        board = chess.Board(fen)
        try:
            move_san = board.san(chess.Move.from_uci(move_uci))
        except ValueError:
            move_san = move_uci
        cache[fen] = (move_uci, move_san)
        return move_uci, move_san

    if fen in cache:
        return cache[fen]

    try:
        engine_path = resolve_engine_path(app_settings)
        depth = app_settings.engine_depth or 12
        with StockfishEngine(engine_path, depth=depth, nodes=app_settings.engine_nodes) as engine:
            _, best_uci = engine.analyse_fen(fen)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Impossible de calculer le meilleur coup : {exc}")
        cache[fen] = (None, None)
        return None, None

    if best_uci:
        board = chess.Board(fen)
        try:
            best_san = board.san(chess.Move.from_uci(best_uci))
        except ValueError:
            best_san = best_uci
    else:
        best_san = None
    cache[fen] = (best_uci, best_san)
    exercise["best_uci"] = best_uci
    return cache[fen]


def apply_filters(
    data: dict[str, pd.DataFrame],
    date_range: tuple[date, date],
    time_classes: list[str],
    colors: list[str],
    openings: list[str],
) -> dict[str, pd.DataFrame]:
    games = sanitize_games(data["games"])
    games = normalize_opening_labels(games)
    if not games.empty:
        start, end = date_range
        mask = (games["utc_date"].dt.date >= start) & (games["utc_date"].dt.date <= end)
        games = games[mask]
        if time_classes:
            games = games[games["time_class"].isin(time_classes)]
        if colors:
            games = games[games["color"].isin(colors)]
        if openings:
            games = games[games["opening"].isin(openings)]

    game_ids = games["game_id"] if not games.empty else pd.Series(dtype=str)

    moves = data["moves"]
    if not moves.empty:
        moves = moves[moves["game_id"].isin(game_ids)]

    blunders = data["blunders"]
    if not blunders.empty:
        blunders = blunders[blunders["game_id"].isin(game_ids)]
        blunders = normalize_opening_labels(blunders)

    openings_summary = normalize_opening_labels(data["openings_summary"])
    if not openings_summary.empty and openings:
        openings_summary = openings_summary[openings_summary["opening"].isin(openings)]

    filtered = data.copy()
    filtered["games"] = games
    filtered["moves"] = moves
    filtered["blunders"] = blunders
    filtered["openings_summary"] = openings_summary
    return filtered


def render_openings_bar(openings: pd.DataFrame) -> None:
    if openings.empty:
        st.info("Pas de données d'ouverture.")
        return
    focus = openings.copy()
    focus["risk_score"] = focus["avg_acpl"] * focus["games"].clip(lower=1)
    filtered = focus[focus["games"] >= 3]
    if filtered.empty:
        filtered = focus
    top = filtered.sort_values("risk_score", ascending=False).head(10)
    chart = (
        alt.Chart(top)
        .mark_bar(color="#f94144")
        .encode(
            x=alt.X("risk_score", title="Indice de risque (ACPL × parties)", sort="descending"),
            y=alt.Y("opening", sort="-x", title="Ouverture"),
            tooltip=[
                "opening",
                alt.Tooltip("avg_acpl", title="ACPL"),
                alt.Tooltip("games", title="Parties"),
                alt.Tooltip("risk_score", title="Indice de risque"),
                "score_pct",
                "blunders_per_game",
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def display_unknown_opening_hint(games: pd.DataFrame, moves: pd.DataFrame) -> None:
    if games.empty or "opening" not in games.columns:
        return
    unknown_games = games[games["opening"] == "Non identifiée"]
    if unknown_games.empty:
        return
    st.warning(
        "Certaines parties restent **Non identifiée (manque de données)**. "
        "Ajoute les séquences ci-dessous à `src/cwa/features/eco_map.json` pour améliorer la détection."
    )
    if moves.empty or "ply" not in moves.columns:
        return
    snippet = moves[moves["game_id"].isin(unknown_games["game_id"])].sort_values("ply")
    if snippet.empty:
        return
    sequences = (
        snippet.groupby("game_id")
        .head(12)
        .groupby("game_id")["san"]
        .apply(lambda seq: " ".join(seq.tolist()))
    )
    for gid, seq in sequences.items():
        st.markdown(f"• `{gid}` : {seq}")
    st.caption("Améliore la détection en ajoutant ces préfixes dans `eco_map.json`.")


def render_trend_chart(moves: pd.DataFrame, games: pd.DataFrame) -> None:
    if moves.empty or games.empty:
        st.info("Pas de coups analysés dans la période filtrée.")
        return
    merged = moves.merge(games[["game_id", "utc_date"]], on="game_id", how="left")
    if "utc_date" not in merged.columns:
        st.info("Les PGN filtrés ne contiennent pas de date (balise UTCDate).")
        return
    merged["utc_date"] = pd.to_datetime(merged["utc_date"], errors="coerce")
    if merged["utc_date"].isna().all():
        st.info("Les dates des parties sont indisponibles pour cette sélection.")
        return
    daily = merged.groupby(merged["utc_date"].dt.date)["abs_delta_cp"].mean().reset_index(name="acpl")
    chart = (
        alt.Chart(daily)
        .mark_line(point=True, color="#1f77b4")
        .encode(x=alt.X("utc_date", title="Date"), y=alt.Y("acpl", title="Perte moyenne (cp)"))
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def render_histogram(moves: pd.DataFrame) -> None:
    if moves.empty:
        st.info("Pas de coups analysés.")
        return
    chart = (
        alt.Chart(moves[moves["delta_cp"].notna()])
        .mark_bar(color="#577590", opacity=0.8)
        .encode(
            x=alt.X("delta_cp", bin=alt.Bin(maxbins=30), title="Perte sur ce coup (Δcp, en cp)"),
            y=alt.Y("count()", title="Nombre de coups"),
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def render_blunders_table(blunders: pd.DataFrame) -> None:
    if blunders.empty:
        st.info("Aucune gaffe détectée.")
        return
    view = blunders[[
        "game_id",
        "opening",
        "phase",
        "san",
        "delta_cp",
        "url",
    ]].copy()
    view = view.rename(
        columns={
            "opening": "Ouverture",
            "phase": "Phase",
            "san": "Coup",
            "delta_cp": "Perte sur ce coup (Δcp, en cp)",
        }
    )
    view["Lien"] = view["url"]
    st.dataframe(view.drop(columns=["url"]), use_container_width=True, hide_index=True)


def render_time_stats(time_stats: pd.DataFrame) -> None:
    if time_stats.empty:
        st.info("Stats de temps indisponibles.")
        return
    st.dataframe(time_stats, use_container_width=True, hide_index=True)


def render_coaching_panel(data: dict[str, pd.DataFrame]) -> None:
    try:
        moves = data["moves"]
        player_moves = moves[moves["played_uci"].notna()] if not moves.empty else pd.DataFrame()
        insights = CoachingInsights.from_data(
            games_df=data["games"],
            player_moves=player_moves,
            blunders=data["blunders"],
            openings=data["openings_summary"],
            non_conversion=data["non_conversion"],
            tactical=data["tactical_events"],
            time_stats=data["time_stats"],
        )
        st.markdown(insights.to_markdown())
    except Exception as exc:  # pragma: no cover
        st.warning(f"Impossible de générer les recommandations: {exc}")


def build_insights(data: dict[str, pd.DataFrame]) -> list[str]:
    items: list[str] = []
    openings = data["openings_summary"]
    if not openings.empty:
        worst = openings.iloc[0]
        items.append(
            f"Ouverture la plus coûteuse : **{worst['opening']}** ({worst['color']}, {worst['time_class']}) avec ACPL {worst['avg_acpl']:.1f}."
        )
    moves = data["moves"]
    if not moves.empty and "phase" in moves.columns:
        phase_stats = moves.groupby("phase")["abs_delta_cp"].mean().sort_values(ascending=False)
        if not phase_stats.empty:
            items.append(
                f"Phase la plus vulnérable : **{phase_stats.index[0]}** (ACPL {phase_stats.iloc[0]:.1f} cp)."
            )
    blunders = data["blunders"]
    if not blunders.empty:
        row = blunders.sort_values("delta_cp").head(1)
        if not row.empty:
            r = row.iloc[0]
            items.append(
                f"Plus grosse erreur : **{r['san']}** ({r['opening']}) avec Δcp {r['delta_cp']:.0f}."
            )
    time_stats = data["time_stats"]
    if not time_stats.empty:
        scramble = time_stats[time_stats["metric"] == "scramble_blunder_rate"]
        if not scramble.empty:
            items.append(
                f"En zeitnot (<20s), taux de blunder : **{float(scramble.iloc[0]['value']):.1f}%**."
            )
    non_conversion = data["non_conversion"]
    if not non_conversion.empty:
        items.append(
            f"Positions gagnantes non converties : **{len(non_conversion)}** épisodes (Δcp min {non_conversion['eval_before_cp'].min():.0f})."
        )
    return items[:5]


def render_coups_typiques(blunders: pd.DataFrame) -> None:
    if blunders.empty:
        st.info("Pas de gaffes dans la période sélectionnée.")
        return

    grouped = (
        blunders.groupby(["opening", "phase", "san", "played_uci", "best_uci"])
        .agg(occurrences=("game_id", "count"), avg_delta=("delta_cp", "mean"))
        .reset_index()
        .sort_values(["occurrences", "avg_delta"], ascending=[False, False])
    )

    display = grouped.rename(
        columns={
            "opening": "Ouverture",
            "phase": "Phase",
            "san": "Coup",
            "occurrences": "Occurrences",
            "avg_delta": "Δcp moyen",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    if display.empty:
        return

    selected = st.selectbox(
        "Sélectionne un motif",
        options=list(grouped.index),
        format_func=lambda idx: " – ".join(
            [grouped.at[idx, "opening"], grouped.at[idx, "phase"].capitalize(), grouped.at[idx, "san"]]
        ),
    )

    sample = (
        blunders[
            (blunders["opening"] == grouped.at[selected, "opening"]) &
            (blunders["phase"] == grouped.at[selected, "phase"]) &
            (blunders["san"] == grouped.at[selected, "san"]) &
            (blunders["played_uci"] == grouped.at[selected, "played_uci"])
        ]
        .head(1)
    )
    if sample.empty:
        st.warning("Impossible de retrouver une partie pour ce motif.")
        return

    sample_row = sample.iloc[0]
    png = render_position(
        fen=sample_row["fen_before"],
        played_uci=sample_row.get("played_uci"),
        best_uci=sample_row.get("best_uci"),
    )
    st.image(png, caption=f"Coup joué : {sample_row['san']}")
    if sample_row.get("url"):
        st.markdown(f"[Voir la partie sur Chess.com]({sample_row['url']})")


def render_training_tab(filtered: dict[str, pd.DataFrame], app_settings, username: str) -> None:
    blunders = filtered["blunders"]
    openings_summary = filtered["openings_summary"]

    st.markdown("""Entraîne-toi sur les positions où l'ACPL est le plus élevé.""")
    if blunders.empty or openings_summary.empty:
        st.info("Pas assez de données pour générer des exercices.")
        return

    top_n = st.slider("Ouvertures ciblées (Top N)", min_value=1, max_value=10, value=3, key="training_top_n")
    per_opening = st.slider("Exercices par ouverture", min_value=1, max_value=10, value=3, key="training_per_opening")

    state_key = f"training_state_{username}"
    signature = (len(blunders), blunders["game_id"].nunique())
    reset_requested = st.button("Recommencer la session")

    if (
        state_key not in st.session_state
        or reset_requested
        or st.session_state[state_key].get("params") != (top_n, per_opening)
        or st.session_state[state_key].get("signature") != signature
    ):
        queue = build_training_queue(blunders, openings_summary, top_n, per_opening)
        st.session_state[state_key] = {
            "queue": queue,
            "index": 0,
            "score": 0,
            "attempted": 0,
            "revealed": False,
            "params": (top_n, per_opening),
            "signature": signature,
        }

    state = st.session_state[state_key]
    queue = state["queue"]

    if not queue:
        st.info("Aucun exercice disponible avec les filtres actuels.")
        return

    expected = top_n * per_opening
    if len(queue) < expected:
        st.info(
            f"Seulement {len(queue)} exercices disponibles (attendu : {expected}). "
            "Élargis les filtres ou enrichis la base."
        )

    if state["index"] >= len(queue):
        st.success(f"Session terminée : {state['score']}/{state['attempted']} réussies.")
        if st.button("Lancer une nouvelle session"):
            queue = build_training_queue(blunders, openings_summary, top_n, per_opening)
            state.update({"queue": queue, "index": 0, "score": 0, "attempted": 0, "revealed": False})
        st.session_state[state_key] = state
        return

    exercise = queue[state["index"]]
    st.write(
        f"Exercice {state['index'] + 1}/{len(queue)} – **{exercise['opening']}** | {exercise['phase'].capitalize()} | Δcp {exercise['delta_cp']:.0f}"
    )
    st.caption("Objectif : trouver le coup recommandé par Stockfish.")

    best_uci, best_san = ensure_best_move(exercise, app_settings) if state["revealed"] else (None, None)
    played_arrow = exercise.get("played_uci") if state["revealed"] else None
    png = render_position(
        fen=exercise.get("fen_before", ""),
        played_uci=played_arrow,
        best_uci=best_uci if state["revealed"] else None,
    )
    st.image(png)
    st.write(f"Coup joué : **{exercise['san']}**")

    if exercise.get("url"):
        st.markdown(f"[Voir la partie sur Chess.com]({exercise['url']})")

    if not state["revealed"]:
        if st.button("Voir la solution"):
            state["revealed"] = True
            best_uci, best_san = ensure_best_move(exercise, app_settings)
    else:
        if best_san or best_uci:
            st.markdown(f"**Coup recommandé** : {best_san or best_uci}")
        else:
            st.info("Le meilleur coup n'a pas pu être déterminé.")

    col_success, col_fail, col_skip = st.columns(3)
    if col_success.button("Réussi", key=f"success_{state['index']}"):
        state["score"] += 1
        state["attempted"] += 1
        state["index"] += 1
        state["revealed"] = False
    if col_fail.button("Échoué", key=f"fail_{state['index']}"):
        state["attempted"] += 1
        state["index"] += 1
        state["revealed"] = False
    if col_skip.button("Passer", key=f"skip_{state['index']}"):
        state["index"] += 1
        state["revealed"] = False

    st.markdown(f"**Score de session** : {state['score']}/{state['attempted']} réussites")
    st.session_state[state_key] = state


def main() -> None:
    st.sidebar.header("Paramètres")
    username = st.sidebar.text_input("Username Chess.com", value=settings.chesscom_username or "")
    if not username:
        st.stop()

    data = load_processed(username)

    if st.sidebar.button("(Re)Analyser"):
        try:
            run_analyze(username=username, settings=settings)
            data = load_processed(username)
            st.sidebar.success("Analyse mise à jour.")
        except Exception as exc:  # pragma: no cover
            st.sidebar.error(f"Analyse impossible: {exc}")

    games = sanitize_games(data["games"])
    if games.empty:
        st.info("Aucune partie analysée. Lance d'abord la CLI `python -m cwa.cli fetch` puis `analyze`.")
        st.stop()

    min_date = games["utc_date"].dt.date.min()
    max_date = games["utc_date"].dt.date.max()
    selected_range = st.sidebar.date_input(
        "Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_range, date):
        selected_range = (selected_range, selected_range)

    time_classes = st.sidebar.multiselect("Cadence", options=sorted(games["time_class"].dropna().unique()))
    colors = st.sidebar.multiselect("Couleur", options=["white", "black"])
    opening_options = sorted(games["opening"].dropna().unique())
    openings = st.sidebar.multiselect("Ouvertures", options=opening_options)

    filtered = apply_filters(data, selected_range, time_classes, colors, openings)

    total_games = int(filtered["games"]["game_id"].nunique()) if not filtered["games"].empty else 0
    distinct_openings = int(filtered["games"]["opening"].nunique()) if not filtered["games"].empty else 0
    st.sidebar.markdown(f"**Parties filtrées** : {total_games}")
    st.sidebar.markdown(f"**Ouvertures distinctes** : {distinct_openings}")
    opening_sum = int(filtered["openings_summary"]["games"].sum()) if not filtered["openings_summary"].empty else 0
    st.sidebar.markdown(f"**Σ games (par ouverture)** : {opening_sum}")
    if opening_sum > total_games and total_games > 0:
        st.sidebar.warning("Σ games > parties distinctes : vérifie les filtres ou relance l'analyse.")
    if total_games <= 3:
        st.sidebar.warning("Échantillon très petit → résultats peu stables.")

    st.title("Chess Weakness Analyzer")
    with st.popover("Comprendre cp, Δcp et ACPL"):
        st.markdown(
            """
            - **cp (centipawns)** : 100 cp = 1 pion d'avantage ou de désavantage.
            - **Δcp** : variation d'évaluation due au coup joué (Δcp < 0 = erreur).
            - **ACPL** : perte moyenne par coup (|Δcp|), indicateur global de précision.
            """
        )

    dashboard_tab, tactics_tab, training_tab = st.tabs(["Tableau de bord", "Coups typiques", "Entraînement"])

    with dashboard_tab:
        kpis = compute_kpis(filtered["player_summary"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Parties", f"{kpis['games']:.0f}")
        col2.metric("Score %", f"{kpis['score_pct']:.1f}")
        col2.markdown(badge_from_threshold("score_pct", kpis["score_pct"]))
        col3.metric("Perte moyenne par coup (ACPL, en cp)", f"{kpis['acpl']:.1f}")
        col3.markdown(badge_from_threshold("acpl", kpis["acpl"]))
        col4.metric("Blunders/100", f"{kpis['blunders']:.1f}")
        col4.markdown(badge_from_threshold("blunders", kpis["blunders"]))

        st.subheader("Ouvertures à risques")
        render_openings_bar(filtered["openings_summary"])
        display_unknown_opening_hint(filtered["games"], filtered["moves"])

        st.subheader("Tendance de la perte moyenne")
        render_trend_chart(filtered["moves"], filtered["games"])

        st.subheader("Distribution des erreurs (Δcp)")
        render_histogram(filtered["moves"])

        st.subheader("Principales gaffes")
        render_blunders_table(filtered["blunders"])

        st.subheader("Insights")
        insights = build_insights(filtered)
        if insights:
            st.markdown("\n".join(f"- {line}" for line in insights))
        else:
            st.info("Pas d'insight sur la période sélectionnée.")

        st.subheader("Stats temps")
        render_time_stats(filtered["time_stats"])

        st.subheader("Ouvertures (table)")
        if not filtered["openings_summary"].empty:
            st.dataframe(
                filtered["openings_summary"].rename(
                    columns={
                        "avg_acpl": "Perte moyenne par coup (cp)",
                        "blunders_per_game": "Blunders/partie",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Pas d'ouvertures sur la période filtrée.")

        st.subheader("Insights & recommandations")
        render_coaching_panel(filtered)

    with tactics_tab:
        render_coups_typiques(filtered["blunders"])

    with training_tab:
        render_training_tab(filtered, settings, username)


if __name__ == "__main__":
    main()
