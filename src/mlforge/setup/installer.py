from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from .doctor import DependencyCheck, doctor_checks


@dataclass(slots=True)
class SetupTask:
    name: str
    title: str
    check_name: str
    install_url: str
    commands: list[list[str]]
    note: str


TASKS = {
    "ollama": SetupTask(
        name="ollama",
        title="Ollama local model runtime",
        check_name="ollama",
        install_url="https://ollama.com/download",
        commands=[],
        note=(
            "Install Ollama from the official page, then run `ollama pull qwen` "
            "or another supported local model."
        ),
    ),
    "firecrawl": SetupTask(
        name="firecrawl",
        title="Firecrawl Local research service",
        check_name="firecrawl-local",
        install_url="https://docs.firecrawl.dev/",
        commands=[],
        note=(
            "Firecrawl-local usually runs through Docker. Use the official Firecrawl "
            "self-hosting instructions, then verify with `saara tools firecrawl-health`."
        ),
    ),
    "docker": SetupTask(
        name="docker",
        title="Docker runtime",
        check_name="docker",
        install_url="https://docs.docker.com/get-docker/",
        commands=[],
        note="Install Docker from the official Docker docs for your operating system.",
    ),
    "data": SetupTask(
        name="data",
        title="Data export extras",
        check_name="pyarrow",
        install_url="https://arrow.apache.org/docs/python/install.html",
        commands=[[sys.executable, "-m", "pip", "install", "saara[data]"]],
        note="Installs optional Parquet, Arrow, and Hugging Face Dataset dependencies.",
    ),
    "agents": SetupTask(
        name="agents",
        title="Agent adapter extras",
        check_name="langchain-core",
        install_url="https://python.langchain.com/docs/introduction/",
        commands=[[sys.executable, "-m", "pip", "install", "saara[agents]"]],
        note="Installs the optional LangChain adapter dependency.",
    ),
    "vllm": SetupTask(
        name="vllm",
        title="vLLM GPU model server",
        check_name="vllm",
        install_url="https://docs.vllm.ai/en/latest/getting_started/installation/",
        commands=[],
        note=(
            "vLLM installation depends on CUDA, GPU, and platform. Follow the official "
            "vLLM install guide, then run a server at `http://localhost:8000/v1`."
        ),
    ),
}


def run_setup(
    targets: list[str],
    yes: bool = False,
    dry_run: bool = False,
    input_func: Callable[[str], str] = input,
) -> list[dict[str, object]]:
    if not targets or "all" in targets:
        targets = ["docker", "ollama", "firecrawl", "data", "agents", "vllm"]
    results: list[dict[str, object]] = []
    checks_by_name = {check.name: check for check in doctor_checks()}
    for target in targets:
        task = TASKS.get(target)
        if task is None:
            raise ValueError(f"Unknown setup target: {target}")
        check = checks_by_name.get(task.check_name)
        results.append(_run_task(task, check, yes, dry_run, input_func))
    return results


def _run_task(
    task: SetupTask,
    check: DependencyCheck | None,
    yes: bool,
    dry_run: bool,
    input_func: Callable[[str], str],
) -> dict[str, object]:
    print(f"\n{task.title}")
    if check:
        status = "OK" if check.ok else "MISSING"
        print(f"Status: {status} - {check.detail}")
        if check.ok:
            return {"target": task.name, "status": "already-installed"}
    print(f"Official install guide: {task.install_url}")
    print(task.note)
    if not task.commands:
        return {"target": task.name, "status": "manual-required", "url": task.install_url}
    for command in task.commands:
        printable = " ".join(command)
        print(f"Proposed command: {printable}")
        if dry_run:
            continue
        if not yes and not _confirm(input_func, f"Run `{printable}`? [y/N]: "):
            return {"target": task.name, "status": "skipped"}
        subprocess.run(command, check=True)
    return {"target": task.name, "status": "installed" if not dry_run else "dry-run"}


def _confirm(input_func: Callable[[str], str], prompt: str) -> bool:
    try:
        answer = input_func(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def host_summary() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "docker": shutil.which("docker") or "",
        "ollama": shutil.which("ollama") or "",
    }
