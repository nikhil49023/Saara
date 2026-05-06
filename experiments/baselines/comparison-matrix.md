# Baseline Comparison Matrix

| System | Primary scope | Local-first | CLI-first | Source-grounded examples | Run artifacts | Validation reports | Label/distill/export lifecycle | Saara positioning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Self-Instruct | Synthetic instruction generation and filtering | No | No | No | Limited by released pipeline | Filtering-focused | Instruction tuning dataset generation | Prior art for generated instruction data; Saara should not claim algorithmic novelty here. |
| Alpaca | Replicable instruction-tuned model and generated data recipe | No | No | No | Dataset/code release | Minimal | Fine-tuning recipe | Prior art for low-cost synthetic instruction data. |
| DataDreamer | Reproducible LLM workflows for generation, fine-tuning, evaluation | Partial | No, Python library first | Possible through user pipelines | Strong | Workflow-dependent | Broad LLM research workflows | Closest systems baseline; Saara differentiates through local CLI UX and built-in artifact conventions. |
| Distilabel | Scalable synthetic data and AI feedback pipelines | Partial | No, Python pipeline first | Possible through user pipelines | Pipeline-dependent | AI feedback focused | Strong generation/judging workflows | Strong engineering baseline; Saara should focus on simpler local artifact workflow. |
| Snorkel | Weak supervision and labeling functions | Partial | No | Not its focus | Strong for labeling workflows | Label quality modeling | Labeling-focused | Prior art for programmatic labeling, not source-grounded LLM dataset generation. |
| DataComp | Dataset curation benchmark and evaluation protocol | No | No | Multimodal dataset curation | Strong benchmark artifacts | Evaluation-focused | Dataset curation benchmark | Supports argument that dataset workflows deserve research focus. |
| Saara | Local-first dataset workflow CLI | Yes | Yes | Built into Firecrawl topic workflow | Built-in run/manifests/reports | Built in | Generate, label, distill, validate, export | Publish as a local-first auditable dataset workflow system, not a new generator. |

## Evaluation Dimensions To Measure

- Setup steps and required services.
- Whether a run can complete offline with local models.
- Whether every output example includes provenance.
- Whether run metadata includes provider/model/config/output paths.
- Validation coverage: schema failures, duplicates, labels, source coverage.
- Ability to export to JSONL/JSON/CSV/Parquet/Arrow/Hugging Face.
- Reproducibility under repeated deterministic runs.
