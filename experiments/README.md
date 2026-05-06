# Saara Research Artifact

This directory turns Saara into a reproducible artifact for a software or systems paper.

The default experiments use the deterministic `mock` provider so reviewers can run the
artifact without external model downloads or API keys. Add Ollama/vLLM and Firecrawl-local
runs when preparing the final paper tables.

## Run

```bash
PYTHONPATH=src python3 scripts/run_research_artifact.py
```

Outputs are written to `experiments/results/`:

- `datasets/`: generated JSONL datasets
- `reports/`: validation reports
- `workspaces/`: Saara run metadata, manifests, and research chunks
- `summary.json`: machine-readable experiment summary
- `summary.md`: paper-ready summary table

## Research Questions

1. Can a local-first CLI generate usable instruction datasets with complete run artifacts?
2. Does the workflow preserve source and run provenance for every generated example?
3. How much operational complexity does Saara remove compared with general LLM workflow libraries?
4. Which quality checks are available today, and which are required before a stronger paper claim?

## Publishable Claim

Saara should be framed as a local-first, auditable workflow system for dataset generation,
labeling, validation, distillation, and export. It should not be framed as a new synthetic
data generation algorithm.

## Required Before Submission

- Add at least one real local-model run with Ollama or vLLM.
- Add at least one Firecrawl-grounded run and report source coverage.
- Add judge/dedup quality gates or explicitly scope them as future work.
- Include the comparison matrix in `baselines/comparison-matrix.md`.
- Archive generated configs, manifests, validation reports, and datasets with the paper.
