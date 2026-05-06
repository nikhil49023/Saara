from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator


CYAN = "\033[38;5;45m"
BLUE = "\033[38;5;81m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"


def is_interactive(stream: object = sys.stdout) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def color(text: str, code: str, enabled: bool | None = None) -> str:
    if enabled is None:
        enabled = is_interactive()
    if not enabled:
        return text
    return f"{code}{text}{RESET}"


def animated_header(title: str, subtitle: str | None = None) -> None:
    if not is_interactive():
        print(title)
        if subtitle:
            print(subtitle)
        return
    line = "=" * min(72, max(24, len(title) + 10))
    print(color(line, BLUE))
    print(color(title, BOLD + CYAN))
    if subtitle:
        print(color(subtitle, DIM))
    print(color(line, BLUE))


def success_pulse(message: str) -> None:
    if not is_interactive():
        print(message)
        return
    for shade in (DIM, BLUE, CYAN, BOLD + CYAN):
        print(f"{CLEAR_LINE}{shade}{message}{RESET}", end="", flush=True)
        time.sleep(0.05)
    print()


@contextmanager
def spinner(message: str) -> Iterator[None]:
    if not is_interactive():
        print(message)
        yield
        return

    stop = threading.Event()
    frames = itertools.cycle(["|", "/", "-", "\\"])

    def run() -> None:
        while not stop.is_set():
            frame = next(frames)
            print(f"{CLEAR_LINE}{CYAN}{frame}{RESET} {message}", end="", flush=True)
            time.sleep(0.08)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.2)
        print(CLEAR_LINE, end="", flush=True)
