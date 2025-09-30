"""Command-line interface for Chess Weakness Analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from cwa.analyzer import AnalysisArtifacts, run_analyze, run_fetch
from cwa.config import get_settings
from cwa.utils import configure_logging, get_logger

console = Console()
LOGGER = get_logger("cli")


def resolve_username(username: Optional[str]) -> str:
    settings = get_settings()
    if username:
        return username
    if settings.chesscom_username:
        return settings.chesscom_username
    raise click.BadParameter("Username required (pass --user or set CHESSCOM_USERNAME)")


def render_analysis_summary(artifacts: AnalysisArtifacts) -> None:
    games = artifacts.games
    moves = artifacts.player_moves
    if games.empty or moves.empty:
        console.print("No data available for summary.")
        return

    table = Table(title="Analyse joueur")
    table.add_column("Segment")
    table.add_column("Valeur")

    table.add_row("Parties", str(len(games)))
    wins = (games["result"].eq("1-0") & games["color"].eq("white")).sum() + (
        games["result"].eq("0-1") & games["color"].eq("black")
    ).sum()
    draws = games["result"].eq("1/2-1/2").sum()
    table.add_row("Score brut", f"{wins} wins / {draws} draws")
    acpl = moves["abs_delta_cp"].mean()
    blunder_rate = 100 * moves["is_blunder"].sum() / max(1, len(moves))
    table.add_row("ACPL", f"{acpl:.1f}")
    table.add_row("Blunders / 100 coups", f"{blunder_rate:.1f}")
    console.print(table)


@click.group(help="Chess Weakness Analyzer (Chess.com)")
@click.pass_context
def cli(ctx: click.Context) -> None:
    configure_logging()
    ctx.ensure_object(dict)


@cli.command(help="Download Chess.com archives into the local cache")
@click.option("--user", type=str, default=None, help="Chess.com username")
@click.option("--max-months", type=int, default=None, help="Limit number of months")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def fetch(user: Optional[str], max_months: Optional[int], force: bool) -> None:
    username = resolve_username(user)
    settings = get_settings()
    console.print(f"[bold]Fetching archives for[/] {username}...")
    results = run_fetch(username=username, settings=settings, max_months=max_months, force=force)
    console.print(f"Fetched {len(results)} monthly archives.")


@cli.command(help="Run engine analysis and produce aggregates")
@click.option("--user", type=str, default=None, help="Chess.com username")
@click.option("--engine-depth", type=int, default=None, help="Engine depth")
@click.option("--engine-nodes", type=int, default=None, help="Engine node limit")
@click.option(
    "--pgn-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    help="Additional PGN file to include",
)
def analyze(
    user: Optional[str],
    engine_depth: Optional[int],
    engine_nodes: Optional[int],
    pgn_file: Optional[Path],
) -> None:
    username = resolve_username(user)
    settings = get_settings()
    console.print(f"[bold]Analyzing games for[/] {username}...")

    artifacts = run_analyze(
        username=username,
        settings=settings,
        engine_depth=engine_depth,
        engine_nodes=engine_nodes,
        pgn_files=[pgn_file] if pgn_file else None,
    )
    render_analysis_summary(artifacts)
    console.print("Outputs stored in data/processed/")


@cli.command(help="Generate coaching report (Markdown/HTML)")
@click.option("--user", type=str, default=None, help="Chess.com username")
@click.option(
    "--out",
    type=click.Path(path_type=Path, dir_okay=False, writable=True, resolve_path=True),
    default=None,
    help="Output path (Markdown by default)",
)
@click.option("--html", is_flag=True, help="Export HTML as well")
@click.option(
    "--pgn-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    help="Additional PGN file to include",
)
def report(
    user: Optional[str],
    out: Optional[Path],
    html: bool,
    pgn_file: Optional[Path],
) -> None:
    username = resolve_username(user)
    settings = get_settings()
    console.print(f"[bold]Building report for[/] {username}...")

    artifacts = run_analyze(
        username=username,
        settings=settings,
        pgn_files=[pgn_file] if pgn_file else None,
    )
    insights = artifacts.coaching
    output_path = out or Path(f"reports/{username}_coaching.md")
    insights.export_markdown(output_path)
    console.print(f"Markdown report saved to {output_path}")
    if html:
        html_path = output_path.with_suffix(".html")
        insights.export_html(html_path)
        console.print(f"HTML report saved to {html_path}")


@cli.command(help="Fetch, analyze, and write reports in a single run")
@click.option("--user", type=str, default=None, help="Chess.com username")
@click.option("--max-months", type=int, default=None, help="Limit number of months to download")
@click.option("--engine-depth", type=int, default=None, help="Engine depth")
@click.option("--engine-nodes", type=int, default=None, help="Engine node limit")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option(
    "--report-out",
    type=click.Path(path_type=Path, dir_okay=False, writable=True, resolve_path=True),
    default=None,
    help="Write Markdown report",
)
@click.option(
    "--pgn-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    help="Additional PGN file to include",
)
def all(
    user: Optional[str],
    max_months: Optional[int],
    engine_depth: Optional[int],
    engine_nodes: Optional[int],
    force: bool,
    report_out: Optional[Path],
    pgn_file: Optional[Path],
) -> None:
    username = resolve_username(user)
    settings = get_settings()
    console.print(f"[bold]Starting full pipeline for[/] {username}")
    run_fetch(username=username, settings=settings, max_months=max_months, force=force)
    artifacts = run_analyze(
        username=username,
        settings=settings,
        engine_depth=engine_depth,
        engine_nodes=engine_nodes,
        pgn_files=[pgn_file] if pgn_file else None,
    )
    render_analysis_summary(artifacts)
    if report_out:
        report_out_path = Path(report_out)
        artifacts.coaching.export_markdown(report_out_path)
        console.print(f"Report saved to {report_out_path}")
    console.print("Pipeline complete.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
