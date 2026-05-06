# Release Plan

Saara should ship in layers so it remains easy to install on any device.

## Versioning

Use semantic versioning:

- `0.1.0-alpha.1`: first public alpha
- `0.1.x`: bug fixes and command polish
- `0.2.0`: PDF ingestion and stronger model routing
- `0.3.0`: richer labeling and distillation workflows
- `1.0.0`: stable workflow/config/schema contracts

## Release Scope for 0.1.0 Alpha

Functional commands:

- `saara`
- `saara splash`
- `saara wizard`
- `saara interactive`
- `saara init`
- `saara models health`
- `saara tools firecrawl-health`
- `saara generate topic`
- `saara label`
- `saara distill`
- `saara validate`
- `saara export`
- `saara run`

Supported base install:

- Python 3.11+
- no required third-party runtime dependencies
- mock provider for smoke tests
- Ollama and vLLM integrations through HTTP
- Firecrawl-local integration through HTTP

Optional extras:

- `data`: Parquet, Arrow, Hugging Face Dataset export
- `pdf`: PDF ingestion dependencies
- `agents`: LangChain adapter
- `all`: all optional local features

## Packaging Strategy

1. Publish Python package with `saara` console script.
2. Recommend `pipx install saara-ai` once package name is final.
3. Recommend `uv tool install saara-ai` for users already using uv.
4. Keep `pip install -e .` for contributors.
5. Build platform-specific PyInstaller binaries after alpha feedback.

## Documentation Requirements

Before alpha release, docs must include:

- installation guide
- quickstart
- command reference
- workflow config reference
- provider setup
- Firecrawl-local setup
- dataset format guide
- troubleshooting
- release notes

## Release Risks

- Package name availability is not confirmed.
- PDF workflow is planned but not implemented.
- Parquet/Arrow/HF exports require optional dependencies.
- Ollama/vLLM behavior depends on local servers and model quality.
- Firecrawl-local requires a running local service.
