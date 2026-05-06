# Saara Release Checklist

## Release Target

Initial release: `0.1.0-alpha.1`

## Before Release

- Confirm package/distribution name availability.
- Confirm repository URL and project metadata.
- Run the full local test suite.
- Run command smoke tests with base dependencies only.
- Run optional data export tests with `.[data]`.
- Verify editable install and isolated install.
- Verify `saara` and `mlforge` console scripts.
- Verify docs build with MkDocs.
- Review README quickstart commands.
- Review generated package metadata with `twine check`.
- Tag release only after artifacts are validated.

## Commands

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python3 -m compileall -q src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
scripts/smoke_cli.sh
python -m build --no-isolation
twine check dist/*
```

## Manual Smoke

```bash
saara
saara splash
saara wizard
saara init
saara models health --provider mock
saara generate topic "release smoke" --samples 2 --provider mock --out /tmp/saara-release.jsonl
saara label /tmp/saara-release.jsonl --labels useful,not-useful --out /tmp/saara-labeled.jsonl
saara distill /tmp/saara-release.jsonl --method dpo --out /tmp/saara-dpo.jsonl
saara validate /tmp/saara-release.jsonl
saara export /tmp/saara-release.jsonl --to json --out /tmp/saara-release.json
```

## Release Channels

- PyPI package for Python users.
- `pipx` install path for CLI users.
- `uv tool install` path for fast isolated installs.
- PyInstaller binaries after the Python package is stable.
- Docker image after GPU/server workflows are added.
