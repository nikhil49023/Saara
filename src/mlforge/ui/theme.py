from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

SAARA_THEME = Theme({
    "brand": "bold cyan",
    "tagline": "dim italic",
    "command": "bold green",
    "path": "blue",
    "error": "bold red",
    "warning": "bold yellow",
    "success": "bold green",
    "info": "cyan",
    "step": "bold blue",
    "marker": "bold white",
})

console = Console(theme=SAARA_THEME)
