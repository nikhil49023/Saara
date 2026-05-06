# Installation Guide

Saara should support four installation modes.

## Recommended User Install

Use `pipx` after Saara is published:

```bash
pipx install saara-ai
saara --help
```

This gives users an isolated environment and a normal executable on PATH.

## Fast User Install With uv

Use `uv tool install` after publishing:

```bash
uv tool install saara-ai
saara --help
```

Use extras when needed:

```bash
uv tool install 'saara-ai[data,pdf]'
```

## Development Install

Use a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
saara --help
```

## Standalone Binary

For users who do not want Python:

```bash
pip install pyinstaller
pyinstaller --onefile --name saara src/mlforge/__main__.py
```

Build separately on Linux, macOS, and Windows. PyInstaller is not a cross-compiler.

## Optional Extras

```bash
pip install 'saara-ai[data]'    # Parquet, Arrow, Hugging Face Dataset
pip install 'saara-ai[pdf]'     # PDF ingestion
pip install 'saara-ai[agents]'  # LangChain adapter
pip install 'saara-ai[all]'     # Everything
```

The base package should stay lightweight.

## Fresh Machine Setup

Saara includes a conservative setup assistant:

```bash
saara doctor
saara setup docker --dry-run
saara setup ollama --dry-run
```

Use `--dry-run` first. It prints every command before anything is changed.

On Debian/Ubuntu Linux, `saara setup docker` installs Docker Engine from Docker's
official apt repository:

- refreshes `apt`
- installs `ca-certificates` and `curl`
- adds Docker's official apt key under `/etc/apt/keyrings`
- adds Docker's official repository
- installs `docker-ce`, `docker-ce-cli`, `containerd.io`, Buildx, and Compose plugin

On Linux, `saara setup ollama` uses the official Ollama installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS, when Homebrew is available:

```bash
saara setup docker --dry-run   # proposes Docker Desktop cask
saara setup ollama --dry-run   # proposes brew install ollama
```

Run for real with confirmation prompts:

```bash
saara setup docker
saara setup ollama
```

Run unattended only when you trust the machine and have reviewed dry-run output:

```bash
saara setup docker ollama --yes
```

Saara does not pull models automatically. After installing Ollama or vLLM, choose a model
for your hardware tier, install it yourself, and then point Saara at that model:

```bash
saara models health --provider ollama --model <your-model>
saara models health --provider vllm --model <your-hf-model> --base-url http://localhost:8000/v1
```

Firecrawl Local requires Docker Compose project configuration and environment variables,
so Saara points to the official self-hosting guide instead of silently generating a service
that may not match your environment:

```bash
saara setup firecrawl --dry-run
saara tools firecrawl-health --base-url http://localhost:3002
```
