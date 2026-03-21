# SAARA CLI

Interactive command-line interface for SAARA AI data engine.

This is an optional companion package to `saara-ai` that provides interactive wizards, commands, and beautiful terminal UI using Rich and Typer.

## Installation

```bash
pip install saara-cli
```

This will automatically install `saara-ai` (the core package) as a dependency.

## Usage

```bash
# Show help
saara --help

# Show version
saara version

# Show info
saara info
```

## Features

- Interactive wizards for dataset creation, training, evaluation, and deployment
- Beautiful Rich-formatted terminal output
- Commands for model management, cloud operations, and RAG operations
- Configuration wizards and guided workflows

## Note

The core `saara-ai` package is a standalone Python library that can be used programmatically without requiring the CLI. The CLI package is completely optional and depends on the core package.

### For Library Usage Only

If you just need to use SAARA as a Python library (in scripts, notebooks, or applications), install only the core package:

```bash
pip install saara-ai
```

Then import and use directly:

```python
from saara import DataPipeline, LLMTrainer, PipelineConfig

config = PipelineConfig()
pipeline = DataPipeline(config)
result = pipeline.process_file("document.pdf", "dataset_name")
```

## License

Proprietary (© 2025-2026 Kilani Sai Nikhil)
