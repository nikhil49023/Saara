# Contributing to Saara

Thanks for helping make Saara better. Contributions of every size are welcome: docs, tests, bug fixes,
new export schemas, providers and workflows. Saara is built for **Hacktoberfest 2026** ("AI belongs to
everyone": building with open-weight models and open-source AI). Note that in 2026 pull requests no longer
count toward Hacktoberfest rewards; contribute because you want to build with us. Issues labelled
`hacktoberfest` are scoped so they can be finished during the event.

## Ground rules
- Be kind. All participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- One focused change per pull request. Spam, low-effort or bulk AI-generated PRs (for example
  whitespace-only edits or mass README tweaks) are closed and labelled `spam`. If an AI assistant helped you,
  say so in the PR and make sure you understand and have tested every line.
- Saara is **general purpose and local-first**: features should work offline with `--provider mock`, and
  must never download models or data without the user's explicit consent.

## Find something to work on
- Issues labelled [`good first issue`](https://github.com/nikhil49023/Saara/labels/good%20first%20issue) are scoped for newcomers.
- Issues labelled [`hacktoberfest`](https://github.com/nikhil49023/Saara/labels/hacktoberfest) are open for Hacktoberfest contributors.
- Comment on an issue before starting so work is not duplicated. For larger ideas, open an issue first.

## Development setup

```bash
git clone https://github.com/<your-username>/Saara.git && cd Saara
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[all,dev]'
python -m pytest -q          # all tests run offline, no GPU or model needed
ruff check src tests         # lint
```

## Making a change
1. Create a branch: `git checkout -b feat/short-description`.
2. Write the change **with tests** in `tests/`. New CLI behaviour needs a test that runs the command
   (see `tests/test_cli_commands.py` and `tests/test_export_engine.py`).
3. Keep the style of the surrounding code: type hints, `from __future__ import annotations`, small
   functions, no new required dependencies (optional extras go in `pyproject.toml`).
4. Update docs when user-facing behaviour changes: `README.md`, `docs/` and `CHANGELOG.md` (under *Unreleased*).
5. Run `python -m pytest -q` and `ruff check src tests`; both must pass.
6. Commit with a clear message, e.g. `feat(export): add axolotl input_output schema`.
7. Open a pull request and fill in the template.

## Project layout

| Path | What lives there |
|---|---|
| `src/mlforge/cli.py` | `saara` command definitions |
| `src/mlforge/workflows/` | generate, label, distill workflows |
| `src/mlforge/datasets/` | validation, file formats, export engine and training schemas |
| `src/mlforge/ingest/` | PDF ingestion (Docling, PyMuPDF, vision OCR fallback) |
| `src/mlforge/providers/` | model providers (mock, Ollama, vLLM, OpenAI-compatible) |
| `src/mlforge/setup/` | doctor, installers, hardware-tiered model setup |
| `experiments/saara-gemma/` | fine-tuning pipeline for Saara's default model |
| `docs/` | user and reference documentation (MkDocs) |

## Adding an export schema
Add a converter to `src/mlforge/datasets/schemas.py` that takes one Saara row and returns the target record,
raising `SchemaError("short_reason")` when a row cannot be represented. Register it in `SCHEMAS`, add tests
in `tests/test_export_engine.py`, and document it in `docs/export.md` and the README table.

## Reporting bugs and security issues
Use the bug report template. Include `saara --version`, your OS and Python version, the exact command and
the output with `--verbose`. For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

## License
By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
