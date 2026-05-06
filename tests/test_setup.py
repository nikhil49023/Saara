from __future__ import annotations

import platform

from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import docker_install_commands, ollama_install_commands, run_setup


def test_doctor_checks_include_core_dependencies() -> None:
    names = {check.name for check in doctor_checks("http://127.0.0.1:9")}

    assert {"python", "docker", "ollama", "firecrawl-local"}.issubset(names)


def test_setup_dry_run_manual_target() -> None:
    results = run_setup(["ollama"], dry_run=True)

    assert results[0]["target"] == "ollama"
    assert results[0]["status"] in {"already-installed", "manual-required", "dry-run"}


def test_docker_install_commands_for_fresh_ubuntu(monkeypatch, tmp_path) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_CODENAME=noble\n', encoding="utf-8")
    monkeypatch.setenv("SAARA_OS_RELEASE", str(os_release))
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    commands = docker_install_commands()
    printed = "\n".join(command.printable() for command in commands)

    assert "download.docker.com/linux/ubuntu" in printed
    assert "docker-ce" in printed
    assert "noble stable" in printed


def test_ollama_install_commands_for_linux(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    commands = ollama_install_commands()

    assert commands
    assert "https://ollama.com/install.sh" in commands[0].printable()
