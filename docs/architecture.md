# Saara Architecture Plan

Saara should be a local-first dataset workflow CLI that installs like a normal developer tool and scales from a laptop to a GPU workstation or server.

## Architecture Decision

Use a small, typed core instead of making LangChain the foundation.

LangChain-compatible adapters can exist at the edge, but the core should own:

- workflow state
- tool calls
- model provider contracts
- artifact provenance
- dataset schemas
- validation reports
- resumability

This keeps installation light and makes the product predictable on low-resource devices. The agent layer should be explicit and bounded: researcher, planner, synthesizer, labeler, judge, deduper, validator, exporter.

## Device Profiles

### Tiny / CPU-Only

Use for small jobs, validation, CSV/JSON conversion, light labeling, and smoke tests.

- Install method: `pipx`, `uv tool install`, or standalone binary.
- Runtime: `mock`, `llama.cpp` server, or remote OpenAI-compatible endpoint.
- Avoid bundled GPU dependencies.
- Default output: JSONL.

### Developer Laptop

Use for normal local workflows.

- Install method: `pipx install saara` or `uv tool install saara`.
- Runtime: Ollama at `http://localhost:11434`.
- Research: Firecrawl-local at `http://localhost:3002`.
- Optional extras: PDF parsing, Parquet/Arrow/HF exports.

### GPU Workstation

Use for large dataset generation, distillation, judging, embeddings, and multimodal parsing.

- Install method: `uv tool install saara[all]` or project venv.
- Runtime: vLLM OpenAI-compatible server at `http://localhost:8000/v1`.
- Optional model workers for embeddings, VLM, judge, and teacher models.
- Store artifacts locally or in S3/MinIO later.

### Team / Server

Use when multiple workflows need queueing or review.

- CLI remains the front door.
- Add optional API server and worker queue.
- Use Postgres for metadata and object storage for artifacts.
- Keep local CLI mode fully functional.

## Core Components

```text
saara CLI
  -> config loader
  -> workflow runner
  -> agent planner
  -> tool registry
  -> model router
  -> artifact store
  -> dataset validator
  -> exporter
```

## Package Layout

```text
src/mlforge/
  cli.py
  core/
    schemas.py
    artifacts.py
    hashing.py
  agents/
    tools.py
    research.py
    langchain_adapter.py
  providers/
    base.py
    ollama.py
    openai_compatible.py
    llama_cpp.py
    transformers_runtime.py
  tools/
    firecrawl.py
    pdf.py
  datasets/
    io.py
    validators.py
    exporters.py
    formats/
  workflows/
    topic_dataset.py
    pdf_dataset.py
    label_dataset.py
    distill.py
  docs/
```

## Workflow Contract

Every workflow should produce:

- `run.json`
- `manifest.json`
- raw model/tool outputs
- intermediate JSONL artifacts
- final dataset
- validation report
- rejected examples with reasons

Every task should record:

- input hashes
- output hashes
- provider and model
- prompt hash
- latency
- token counts when available
- retry count
- validation result

## Agentic Flow

Do not use an unbounded autonomous agent for production dataset generation. Use bounded agents with declared tools:

```text
ResearchAgent
  -> firecrawl_local.search
  -> firecrawl_local.scrape
  -> DocumentChunk[]

DatasetPlanner
  -> schema + format + quality gates

Synthesizer
  -> candidate DatasetExample[]

Judge
  -> quality scores + rejection reasons

Validator
  -> ValidationReport

Exporter
  -> JSONL / JSON / CSV / TSV / Parquet / Arrow / HF
```

## Tool Layer

The tool layer should be framework-neutral:

```python
class AgentTool:
    name: str
    description: str
    def invoke(self, call: ToolCall) -> ToolResult: ...
```

Firecrawl-local is the first required tool:

- `search(query, limit)`
- `scrape(url)`

LangChain should be optional through an adapter. This avoids making every install pull in a larger agent stack while keeping compatibility for users who already use LangChain.

## Model Runtime Layer

Use capability routing, not hardcoded model names:

```yaml
models:
  generator:
    provider: ollama
    model: qwen
  vision_parser:
    provider: ollama
    model: qwen-vl
  judge:
    provider: vllm
    model: Qwen/Qwen3
  embedding:
    provider: ollama
    model: nomic-embed-text
```

Runtime recommendations:

- Ollama: default local developer experience.
- vLLM: high-throughput GPU server through OpenAI-compatible APIs.
- llama.cpp: CPU/GGUF/edge fallback.
- Transformers: direct research and fine-tuning path, optional extra.

## Installation Strategy

Ship in layers:

1. Python package with console script: `mlforge`.
2. `pipx install saara` for normal users.
3. `uv tool install saara` for fast isolated installs.
4. Optional extras: `saara[data]`, `saara[pdf]`, `saara[agents]`, `saara[all]`.
5. Standalone PyInstaller binaries for users without Python.
6. Docker image for server/GPU environments.

The base install must have minimal dependencies. Heavy libraries belong in extras.

## Documentation Architecture

Use Markdown docs with MkDocs Material:

```text
docs/
  index.md
  install.md
  quickstart.md
  user-guide.md
  architecture.md
  workflows/
    topic-to-dataset.md
    pdf-to-dataset.md
    labeling.md
    distillation.md
  reference/
    cli.md
    config.md
    formats.md
    providers.md
    tools.md
  troubleshooting.md
```

Documentation must include runnable examples and expected outputs.

## Research Notes

- `uv` provides standalone installers, package-manager installs, PyPI distribution, and isolated tool installation patterns: https://docs.astral.sh/uv/getting-started/installation/
- `pipx` is built for isolated Python application installs and works on macOS, Linux, and Windows: https://pipx.pypa.io/stable/how-to/install-pipx/
- PyInstaller can bundle Python applications and dependencies so users can run without installing Python, but builds are platform-specific: https://pyinstaller.org/en/stable/index.html
- Ollama exposes local HTTP generation APIs and supports structured output via JSON or JSON schema: https://docs.ollama.com/api/generate
- vLLM exposes an OpenAI-compatible HTTP server for completions, chat, embeddings, scoring, and related APIs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- MkDocs is a Markdown-based static docs generator that builds static HTML and includes a preview server: https://www.mkdocs.org/
