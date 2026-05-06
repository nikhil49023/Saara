# Installation Guide

Saara should support four installation modes.

## Recommended User Install

Use `pipx` after Saara is published:

```bash
pipx install saara
saara --help
```

This gives users an isolated environment and a normal executable on PATH.

## Fast User Install With uv

Use `uv tool install` after publishing:

```bash
uv tool install saara
saara --help
```

Use extras when needed:

```bash
uv tool install 'saara[data,pdf]'
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
pip install 'saara[data]'    # Parquet, Arrow, Hugging Face Dataset
pip install 'saara[pdf]'     # PDF ingestion
pip install 'saara[agents]'  # LangChain adapter
pip install 'saara[all]'     # Everything
```

The base package should stay lightweight.
