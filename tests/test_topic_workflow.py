from __future__ import annotations

from pathlib import Path

from mlforge.core.artifacts import ArtifactStore
from mlforge.datasets.io import load_rows
from mlforge.datasets.validators import validate_dataset
from mlforge.providers.mock import MockProvider
from mlforge.workflows.topic_dataset import TopicDatasetWorkflow


def test_topic_workflow_generates_valid_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "dataset.jsonl"
    workflow = TopicDatasetWorkflow(
        topic="robotics motion planning",
        samples=3,
        provider=MockProvider(),
        output_path=output,
        store=ArtifactStore(tmp_path / ".mlforge"),
    )

    manifest = workflow.run()

    assert manifest["examples"] == 3
    assert output.exists()
    rows = load_rows(output)
    assert len(rows) == 3
    report = validate_dataset(output)
    assert report.invalid_examples == 0
