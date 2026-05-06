from __future__ import annotations

from pathlib import Path

from mlforge.datasets.io import load_rows
from mlforge.providers.mock import MockProvider
from mlforge.workflows.distill import DistillWorkflow
from mlforge.workflows.label_dataset import LabelDatasetWorkflow
from mlforge.workflows.topic_dataset import TopicDatasetWorkflow


def test_label_and_distill_workflows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    labeled = tmp_path / "labeled.jsonl"
    distilled = tmp_path / "distilled.jsonl"
    provider = MockProvider()

    TopicDatasetWorkflow(
        topic="release readiness",
        samples=2,
        provider=provider,
        output_path=source,
    ).run()

    label_manifest = LabelDatasetWorkflow(
        input_path=source,
        output_path=labeled,
        provider=provider,
        labels=["accepted", "rejected"],
    ).run()

    assert label_manifest["examples"] == 2
    rows = load_rows(labeled)
    assert rows[0]["labels"]["label"] in {"accepted", "rejected"}

    distill_manifest = DistillWorkflow(
        input_path=source,
        output_path=distilled,
        provider=provider,
        method="dpo",
    ).run()

    assert distill_manifest["examples"] == 2
    dpo_rows = load_rows(distilled)
    assert {"prompt", "chosen", "rejected"}.issubset(dpo_rows[0])
