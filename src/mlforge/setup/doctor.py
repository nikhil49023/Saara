from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from mlforge.tools.firecrawl import FirecrawlClient


@dataclass(slots=True)
class DependencyCheck:
    name: str
    ok: bool
    required: bool
    detail: str
    install_url: str | None = None
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def doctor_checks(firecrawl_url: str = "http://localhost:3002") -> list[DependencyCheck]:
    return [
        _python_check(),
        _command_check(
            name="docker",
            required=False,
            command="docker --version",
            install_url="https://docs.docker.com/get-docker/",
        ),
        _command_check(
            name="ollama",
            required=False,
            command="ollama --version",
            install_url="https://ollama.com/download",
        ),
        _firecrawl_check(firecrawl_url),
        _python_module_check(
            name="pyarrow",
            required=False,
            install_url="https://arrow.apache.org/docs/python/install.html",
            command="pip install 'saara-ai[data]'",
        ),
        _python_module_check(
            name="datasets",
            required=False,
            install_url="https://huggingface.co/docs/datasets/installation",
            command="pip install 'saara-ai[data]'",
        ),
    ]


def _python_check() -> DependencyCheck:
    import sys

    ok = sys.version_info >= (3, 11)
    version = ".".join(str(part) for part in sys.version_info[:3])
    return DependencyCheck(
        name="python",
        ok=ok,
        required=True,
        detail=f"Python {version}",
        install_url="https://www.python.org/downloads/",
    )


def _command_check(name: str, required: bool, command: str, install_url: str) -> DependencyCheck:
    binary = command.split()[0]
    if not shutil.which(binary):
        return DependencyCheck(
            name=name,
            ok=False,
            required=required,
            detail=f"{binary} not found on PATH",
            install_url=install_url,
            command=command,
        )
    try:
        result = subprocess.run(
            command.split(),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        return DependencyCheck(name, False, required, str(exc), install_url, command)
    output = (result.stdout or result.stderr).strip()
    return DependencyCheck(name, result.returncode == 0, required, output, install_url, command)


def _firecrawl_check(base_url: str) -> DependencyCheck:
    health = FirecrawlClient(base_url).health()
    return DependencyCheck(
        name="firecrawl-local",
        ok=bool(health.get("ok")),
        required=False,
        detail=str(health.get("base_url") or health.get("error") or "unknown"),
        install_url="https://docs.firecrawl.dev/",
        command=f"saara tools firecrawl-health --base-url {base_url}",
    )


def _python_module_check(
    name: str,
    required: bool,
    install_url: str,
    command: str,
) -> DependencyCheck:
    ok = importlib.util.find_spec(name) is not None
    return DependencyCheck(
        name=name,
        ok=ok,
        required=required,
        detail="installed" if ok else "not installed",
        install_url=install_url,
        command=command,
    )
