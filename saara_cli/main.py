"""
SAARA CLI Main Entry Point

Interactive command-line interface for SAARA AI.
"""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="saara",
    help="🧠 SAARA - Autonomous Document-to-LLM Data Factory",
    add_completion=False,
    no_args_is_help=True
)


@app.command()
def version():
    """Show SAARA version."""
    from saara import __version__
    typer.echo(f"SAARA version {__version__}")


@app.command()
def info():
    """Show SAARA information."""
    from saara import __version__, __copyright__
    typer.echo(f"""
╭─ SAARA AI ────────────────────╮
│ Version: {__version__}              │
│ {__copyright__} │
│                               │
│ Documentation & Examples:      │
│ https://github.com/nikhil49023 │
╰───────────────────────────────╯
""")


def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user")
        raise typer.Exit(code=130)


if __name__ == "__main__":
    main()
