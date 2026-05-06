from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.status import Status

from .theme import console


def is_interactive(stream: object = None) -> bool:
    if stream is None:
        return console.is_terminal
    return bool(getattr(stream, "isatty", lambda: False)())


def color(text: str, style: str) -> str:
    return f"[{style}]{text}[/]"


def animated_header(title: str, subtitle: str | None = None) -> None:
    console.rule(f"[brand]{title}[/]")
    if subtitle:
        console.print(f"[tagline]{subtitle}[/]", justify="center")


def success_pulse(message: str) -> None:
    console.print(f"[success]✔[/] {message}")


@contextmanager
def spinner(message: str) -> Iterator[None]:
    if not is_interactive():
        console.print(message)
        yield
        return

    with console.status(f"[info]{message}...[/]", spinner="dots") as status:
        yield
