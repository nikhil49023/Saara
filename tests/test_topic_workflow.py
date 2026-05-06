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


def test_topic_workflow_generates_pretraining_text(tmp_path: Path) -> None:
    output = tmp_path / "pretraining.jsonl"
    workflow = TopicDatasetWorkflow(
        topic="dataset curation",
        samples=1,
        provider=MockProvider(),
        output_path=output,
        dataset_type="pretraining",
        store=ArtifactStore(tmp_path / "artifact"),
    )

    workflow.run()

    row = load_rows(output)[0]
    assert row["task_type"] == "pretraining_text"
    assert row["output"]["text"]
    assert row["metadata"]["dataset_type"] == "pretraining"


def test_topic_workflow_generates_reasoning_and_tool_calling(tmp_path: Path) -> None:
    output = tmp_path / "tools.jsonl"
    workflow = TopicDatasetWorkflow(
        topic="source grounded qa",
        samples=1,
        provider=MockProvider(),
        output_path=output,
        dataset_type="tool-calling",
        include_reasoning=True,
        include_tool_calls=True,
        store=ArtifactStore(tmp_path / "artifact"),
    )

    workflow.run()

    row = load_rows(output)[0]
    assert row["task_type"] == "tool_calling"
    assert row["output"]["reasoning"]
    assert row["tool_calls"]
    assert row["tools"]
