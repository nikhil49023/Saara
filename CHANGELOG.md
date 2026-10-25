# Changelog

All notable changes to Saara are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.8.0] - 2026-10-25 (Hacktoberfest 2026)

Version 1.7.0 was already used by the earlier SAARA Desktop GitHub release, so this release is 1.8.0.

### Added
- **Export engine**: `saara export --schema` converts datasets to `openai-chat`, `sharegpt`, `alpaca`, `chatml`,
  `vertex-gemini`, `dpo`, `text` or native `saara` rows, with deterministic group-aware splits (`--split`,
  `--group-by`), sharding (`--shard-size`), gzip compression, a sha256 manifest, a Hugging Face dataset card and
  a LLaMA-Factory `dataset_info.json`. Rows a schema cannot represent are dropped with a counted reason, or fail
  with `--strict`.
- **PDF ingestion**: `saara ingest pdf` with Docling layout parsing, PyMuPDF text extraction and a Gemma 4 vision
  OCR fallback for scanned or sparse pages.
- **Local model setup**: `saara models hardware|setup|list|remove` with hardware tiering (Gemma 4 E2B / E4B / 12B),
  an explicit consent gate before any download, SHA-256 verification and offline `--model-path` registration.
- Wizard step for local model setup.
- `--provider gemini`: hosted Gemma 4 / Gemini through the official `google-genai` SDK, with image parts for
  vision OCR and Gemma 4 thinking level (`pip install 'saara-ai[gemini]'`).
- `saara --version`, a quick-start section in `saara --help`, and examples in `saara export --help`.
- **saara-gemma** fine-tuning pipeline (`experiments/saara-gemma/`) for Gemma 4 E4B on public datasets, with
  evaluation and auto-generated release docs. See `docs/saara-gemma.md`.
- Contributor docs: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates, CI workflow.

### Fixed
- saara-gemma: the Hermes function-calling parser dropped every row (the prompt mentions `<tools> </tools>` in prose
  before the real block); Hermes data is now included.
- `pip install saara-ai` (no extras) crashed at startup: `rich` is now a core dependency.
- Parquet, Arrow and HF exports crashed on rows with empty nested fields (for example `labels: {}`).
- `distill --method dpo` used a constant refusal string as every `rejected` answer, which teaches a model to
  avoid admitting uncertainty. Negatives are now another example's answer, or a truncated answer.
- Error messages lost text in square brackets (rich markup); file-not-found and missing-dependency errors
  are now readable, and Ctrl-C exits cleanly with code 130.
- The package version reported at runtime now comes from the installed package metadata.

## [1.6.9] - 2026-05-06
- Published the Saara CLI and research artifact.
