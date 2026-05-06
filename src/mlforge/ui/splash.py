from __future__ import annotations

import sys
import time

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text

from .theme import console

BRAND_NAME = "Saara"
TAGLINE = "Local-first dataset agents for research, labeling, and distillation"

LOGO = r"""
   _____                       
  / ___/____ _____ __________ _
  \__ \/ __ `/ __ `/ ___/ __ `/
 ___/ / /_/ / /_/ / /  / /_/ / 
/____/\__,_/\__,_/_/   \__,_/  
"""

def splash_text() -> str:
    """Returns the plain text version of the splash screen."""
    return f"{LOGO}\n{TAGLINE}"

def render_splash(animated: bool = True, seconds: float = 1.4, stream: object = sys.stdout) -> None:
    if not animated or not getattr(stream, "isatty", lambda: False)():
        _print_static_splash()
        return

    _animate_splash(seconds)


def _print_static_splash() -> None:
    logo_text = Text(LOGO, style="brand")
    console.print(Align.center(logo_text))
    console.print(Align.center(Text(TAGLINE, style="tagline")))
    console.print()
    _print_getting_started()


def _animate_splash(seconds: float) -> None:
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[brand]Loading Saara...[/]", total=100)
        
        while not progress.finished:
            progress.update(task, advance=5)
            time.sleep(seconds / 20)
            
    console.clear()
    _print_static_splash()


def _print_getting_started() -> None:
    commands = [
        "saara wizard",
        "saara doctor",
        "saara setup",
        "saara init",
        'saara generate topic "robotics motion planning" --samples 20 --provider mock',
        "saara validate .mlforge/datasets/robotics-motion-planning.jsonl",
        "saara --help",
    ]
    
    getting_started = Text("Get started\n", style="bold white")
    for cmd in commands:
        getting_started.append(f"$ ", style="dim")
        getting_started.append(f"{cmd}\n", style="command")
    
    console.print(Align.center(getting_started))
