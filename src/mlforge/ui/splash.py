from __future__ import annotations

import os
import shutil
import sys
import time

BRAND_NAME = "Saara"
TAGLINE = "Local-first dataset agents for research, labeling, and distillation"

LOGO = [
    "   _____                       ",
    "  / ___/____ _____ __________ _",
    "  \\__ \\/ __ `/ __ `/ ___/ __ `/",
    " ___/ / /_/ / /_/ / /  / /_/ / ",
    "/____/\\__,_/\\__,_/_/   \\__,_/  ",
]

PALETTE = [
    "\033[38;5;45m",
    "\033[38;5;81m",
    "\033[38;5;117m",
    "\033[38;5;153m",
    "\033[38;5;159m",
]
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def supports_color(stream: object = sys.stdout) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def splash_text(color: bool = True, frame: int = 0) -> str:
    width = shutil.get_terminal_size((88, 24)).columns
    lines: list[str] = []
    for index, line in enumerate(LOGO):
        prefix = PALETTE[(index + frame) % len(PALETTE)] if color else ""
        lines.append(_center(f"{prefix}{line}{RESET if color else ''}", width))
    subtitle = f"{BOLD if color else ''}{TAGLINE}{RESET if color else ''}"
    lines.append("")
    lines.append(_center(subtitle, width))
    lines.append("")
    lines.extend(_getting_started_lines(width, color))
    return "\n".join(lines)


def render_splash(animated: bool = True, seconds: float = 1.4, stream: object = sys.stdout) -> None:
    color = supports_color(stream)
    if not animated or not getattr(stream, "isatty", lambda: False)():
        print(splash_text(color=color, frame=0), file=stream)
        return

    frames = max(1, int(seconds / 0.08))
    for frame in range(frames):
        print("\033[2J\033[H", end="", file=stream)
        print(splash_text(color=color, frame=frame), file=stream)
        print("", file=stream)
        print(_center(_progress(frame + 1, frames, color), shutil.get_terminal_size((88, 24)).columns), file=stream)
        stream.flush()
        time.sleep(0.08)


def _progress(current: int, total: int, color: bool) -> str:
    cells = 28
    filled = int(cells * current / total)
    bar = "#" * filled + "-" * (cells - filled)
    if color:
        return f"\033[38;5;45m[{bar}]\033[0m"
    return f"[{bar}]"


def _center(text: str, width: int) -> str:
    plain_len = _visible_len(text)
    padding = max(0, (width - plain_len) // 2)
    return " " * padding + text


def _getting_started_lines(width: int, color: bool) -> list[str]:
    title = f"{BOLD if color else ''}Get started{RESET if color else ''}"
    commands = [
        "saara wizard",
        "saara doctor",
        "saara setup",
        "saara init",
        'saara generate topic "robotics motion planning" --samples 20 --provider mock',
        "saara validate .mlforge/datasets/robotics-motion-planning.jsonl",
        "saara --help",
    ]
    lines = [_center(title, width)]
    for command in commands:
        prefix = f"{DIM if color else ''}$ {RESET if color else ''}"
        lines.append(_center(f"{prefix}{command}", width))
    return lines


def _visible_len(text: str) -> int:
    length = 0
    index = 0
    while index < len(text):
        if text[index : index + 2] == "\033[":
            end = text.find("m", index)
            if end == -1:
                break
            index = end + 1
            continue
        length += 1
        index += 1
    return length
