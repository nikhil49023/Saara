# Paper Outline

## Working Title

Saara: A Local-First CLI for Auditable Dataset Generation, Labeling, Validation, and Distillation

## Abstract Sketch

Large language models are increasingly used to generate, label, and transform datasets, but
many workflows depend on ad hoc scripts, remote services, and incomplete run provenance.
We present Saara, a local-first command-line workflow system for source-grounded dataset
generation and dataset lifecycle operations. Saara combines bounded web research tools,
local model provider routing, canonical dataset examples, artifact manifests, validation
reports, and exports into reproducible workflows that run on developer laptops and scale to
local GPU servers. We evaluate Saara on deterministic smoke experiments and local-model
source-grounded generation tasks, comparing artifact completeness and workflow complexity
against existing synthetic data workflow tools.

## Contributions

- A local-first CLI architecture for dataset generation and transformation workflows.
- A bounded research-agent pattern that treats retrieved web content as data and records source provenance.
- A canonical dataset artifact model with manifests, validation reports, run metadata, and exports.
- A reproducible artifact package with configs, generated datasets, validation reports, and comparison tables.

## Experiments

1. Deterministic mock-provider runs for reproducible artifact verification.
2. Local Ollama/vLLM runs for realistic model-generated instruction datasets.
3. Firecrawl-grounded runs measuring source coverage and generated example validity.
4. Baseline comparison against DataDreamer, Distilabel, and a hand-written script.

## Metrics

- Total examples.
- Valid examples.
- Invalid examples.
- Duplicate examples.
- Source coverage.
- Label distribution for labeled datasets.
- Artifact completeness: config, run metadata, research chunks, dataset, validation report, manifest.
- Setup complexity: number of commands and external services.

## Claims To Avoid

- Do not claim a new synthetic data generation algorithm.
- Do not claim better dataset quality without downstream evaluation.
- Do not claim full reproducibility for remote closed models unless prompts, model versions, and raw outputs are archived.

## Stronger Submission Path

For a workshop or software paper, the current artifact plus mock and local-model runs is enough.
For a stronger systems or dataset venue, add judge scoring, deduplication, rejected examples,
and downstream fine-tuning/evaluation results.
