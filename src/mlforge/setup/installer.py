from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .doctor import DependencyCheck, doctor_checks


@dataclass(slots=True)
class SetupCommand:
    command: list[str] | str
    description: str
    shell: bool = False

    def printable(self) -> str:
        if isinstance(self.command, str):
            return self.command
        return " ".join(self.command)


@dataclass(slots=True)
class SetupTask:
    name: str
    title: str
    check_name: str
    install_url: str
    note: str
    command_factory: Callable[[], list[SetupCommand]]


def _empty_commands() -> list[SetupCommand]:
    return []


TASKS = {
    "docker": SetupTask(
        name="docker",
        title="Docker runtime",
        check_name="docker",
        install_url="https://docs.docker.com/engine/install/",
        note=(
            "Installs Docker Engine using the official Docker apt repository on Debian/Ubuntu. "
            "macOS and Windows should use Docker Desktop from the official Docker docs."
        ),
        command_factory=lambda: docker_install_commands(),
    ),
    "ollama": SetupTask(
        name="ollama",
        title="Ollama local model runtime",
        check_name="ollama",
        install_url="https://ollama.com/download",
        note=(
            "Installs Ollama from the official installer on Linux, or Homebrew on macOS. "
            "Saara never pulls models automatically; choose one later based on available CPU/GPU memory."
        ),
        command_factory=lambda: ollama_install_commands(),
    ),
    "firecrawl": SetupTask(
        name="firecrawl",
        title="Firecrawl Local research service",
        check_name="firecrawl-local",
        install_url="https://docs.firecrawl.dev/contributing/self-host",
        note=(
            "Firecrawl Local is a Docker Compose service. Saara verifies Docker first and "
            "then prints the official self-hosting path because Firecrawl requires project "
            "configuration and environment variables."
        ),
        command_factory=_empty_commands,
    ),
    "data": SetupTask(
        name="data",
        title="Data export extras",
        check_name="pyarrow",
        install_url="https://arrow.apache.org/docs/python/install.html",
        command_factory=lambda: [
            SetupCommand([sys.executable, "-m", "pip", "install", "saara-ai[data]"], "Install data extras")
        ],
        note="Installs optional Parquet, Arrow, and Hugging Face Dataset dependencies.",
    ),
    "agents": SetupTask(
        name="agents",
        title="Agent adapter extras",
        check_name="langchain-core",
        install_url="https://python.langchain.com/docs/introduction/",
        command_factory=lambda: [
            SetupCommand(
                [sys.executable, "-m", "pip", "install", "saara-ai[agents]"],
                "Install LangChain adapter extras",
            )
        ],
        note="Installs the optional LangChain adapter dependency.",
    ),
    "vllm": SetupTask(
        name="vllm",
        title="vLLM GPU model server",
        check_name="vllm",
        install_url="https://docs.vllm.ai/en/latest/getting_started/installation/",
        command_factory=_empty_commands,
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


def docker_install_commands() -> list[SetupCommand]:
    system = platform.system().lower()
    if system == "linux" and _linux_id() in {"ubuntu", "debian"}:
        distro = _linux_id()
        codename = _linux_codename()
        if not codename:
            return []
        repo = (
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] "
            f"https://download.docker.com/linux/{distro} {codename} stable"
        )
        return [
            SetupCommand(["sudo", "apt-get", "update"], "Refresh package index"),
            SetupCommand(
                ["sudo", "apt-get", "install", "-y", "ca-certificates", "curl"],
                "Install repository prerequisites",
            ),
            SetupCommand(["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"], "Create apt keyring dir"),
            SetupCommand(
                [
                    "sudo",
                    "curl",
                    "-fsSL",
                    f"https://download.docker.com/linux/{distro}/gpg",
                    "-o",
                    "/etc/apt/keyrings/docker.asc",
                ],
                "Download Docker apt key",
            ),
            SetupCommand(["sudo", "chmod", "a+r", "/etc/apt/keyrings/docker.asc"], "Allow apt to read Docker key"),
            SetupCommand(
                f"printf '%s\\n' '{repo}' | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null",
                "Add Docker apt repository",
                shell=True,
            ),
            SetupCommand(["sudo", "apt-get", "update"], "Refresh package index with Docker repository"),
            SetupCommand(
                [
                    "sudo",
                    "apt-get",
                    "install",
                    "-y",
                    "docker-ce",
                    "docker-ce-cli",
                    "containerd.io",
                    "docker-buildx-plugin",
                    "docker-compose-plugin",
                ],
                "Install Docker Engine and Compose plugin",
            ),
        ]
    if system == "darwin" and shutil.which("brew"):
        return [SetupCommand(["brew", "install", "--cask", "docker"], "Install Docker Desktop with Homebrew")]
    return []


def ollama_install_commands() -> list[SetupCommand]:
    system = platform.system().lower()
    if system == "linux":
        return [
            SetupCommand(
                "curl -fsSL https://ollama.com/install.sh | sh",
                "Install Ollama using the official Linux installer",
                shell=True,
            )
        ]
    if system == "darwin" and shutil.which("brew"):
        return [SetupCommand(["brew", "install", "ollama"], "Install Ollama with Homebrew")]
    return []


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

    commands = task.command_factory()
    if not commands:
        return {
            "target": task.name,
            "status": "manual-required",
            "url": task.install_url,
            "reason": "No safe automated installer is available for this platform or target.",
        }

    for command in commands:
        printable = command.printable()
        print(f"Step: {command.description}")
        print(f"Command: {printable}")
        if dry_run:
            continue
        if not yes and not _confirm(input_func, f"Run this command? [y/N]: "):
            return {"target": task.name, "status": "skipped"}
        subprocess.run(command.command, shell=command.shell, check=True)
    return {"target": task.name, "status": "installed" if not dry_run else "dry-run"}


def _confirm(input_func: Callable[[str], str], prompt: str) -> bool:
    try:
        answer = input_func(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _linux_id() -> str:
    return _os_release().get("ID", "").lower()


def _linux_codename() -> str:
    release = _os_release()
    return release.get("VERSION_CODENAME") or release.get("UBUNTU_CODENAME") or ""


def _os_release() -> dict[str, str]:
    path = Path(os.environ.get("SAARA_OS_RELEASE", "/etc/os-release"))
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def host_summary() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "docker": shutil.which("docker") or "",
        "ollama": shutil.which("ollama") or "",
    }
