from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mlforge.cli import main
from mlforge.setup.model import (
    MODEL_CATALOG,
    HardwareDetector,
    HardwareProfile,
    ModelManager,
)


def test_hardware_detector():
    profile = HardwareDetector.detect()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb > 0
    assert profile.available_ram_gb > 0
    assert profile.free_disk_gb >= 0
    assert profile.recommended_tier in {"tier_1", "tier_2", "tier_3"}


def test_model_manager_register_and_remove(tmp_path: Path):
    manager = ModelManager(root_dir=tmp_path / "models")
    dummy_model = tmp_path / "dummy_gemma.gguf"
    dummy_model.write_bytes(b"GGUF_TEST_HEADER_DATA_1234567890")

    # Register local model
    entry = manager.register_local_model("test-gemma", dummy_model)
    assert entry["name"] == "test-gemma"
    assert entry["path"] == str(dummy_model.resolve())
    assert len(entry["sha256"]) == 64

    # Verify listing
    models = manager.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "test-gemma"

    # Remove model
    removed = manager.remove_model("test-gemma")
    assert removed is True
    assert len(manager.list_models()) == 0


def test_cli_models_hardware():
    # Should run and exit 0 without crashing
    main(["models", "hardware"])


def test_cli_models_setup_dry_run(capsys):
    main(["models", "setup", "--dry-run"])
    captured = capsys.readouterr()
    assert "Saara Model Setup" in captured.out
    assert "Dry-run mode" in captured.out


def test_cli_models_setup_with_local_path(tmp_path: Path):
    dummy_model = tmp_path / "offline_gemma.gguf"
    dummy_model.write_bytes(b"OFFLINE_GGUF_MODEL_DATA")

    with patch.object(Path, "home", return_value=tmp_path):
        main(["models", "setup", "--model-path", str(dummy_model), "--model", "offline-gemma"])

    manager = ModelManager(root_dir=tmp_path / ".mlforge" / "models")
    models = manager.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "offline-gemma"


def test_cli_models_setup_consent_declined(capsys):
    with patch("builtins.input", return_value="n"):
        main(["models", "setup"])
    captured = capsys.readouterr()
    assert "Consent not granted" in captured.out
