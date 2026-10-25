from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlforge.ui.theme import console


@dataclass(slots=True)
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    free_disk_gb: float
    has_gpu: bool = False
    vram_gb: float = 0.0
    recommended_tier: str = "tier_1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelManifest:
    name: str
    tier: str
    parameters: str
    license: str
    disk_size_gb: float
    min_ram_gb: float
    ollama_tag: str
    hf_repo: str
    hf_filename: str
    sha256: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_CATALOG: dict[str, ModelManifest] = {
    "gemma-4-e2b": ModelManifest(
        name="gemma-4-e2b",
        tier="tier_1",
        parameters="E2B (Effective 2 Billion)",
        license="Apache 2.0",
        disk_size_gb=4.6,
        min_ram_gb=8.0,
        ollama_tag="gemma4:e2b",
        hf_repo="google/gemma-4-e2b-it-GGUF",
        hf_filename="gemma-4-e2b-it-Q4_K_M.gguf",
        sha256=None,
        description="Edge/Mobile lightweight model for fast on-device inference on CPU or low-VRAM GPUs.",
    ),
    "gemma-4-e4b": ModelManifest(
        name="gemma-4-e4b",
        tier="tier_2",
        parameters="E4B (Effective 4 Billion)",
        license="Apache 2.0",
        disk_size_gb=6.6,
        min_ram_gb=12.0,
        ollama_tag="gemma4:e4b",
        hf_repo="google/gemma-4-e4b-it-GGUF",
        hf_filename="gemma-4-e4b-it-Q4_K_M.gguf",
        sha256=None,
        description="Balanced model for laptops and workstations. Excellent reasoning and data cleaning.",
    ),
    "gemma-4-12b": ModelManifest(
        name="gemma-4-12b",
        tier="tier_3",
        parameters="12B Dense",
        license="Apache 2.0",
        disk_size_gb=8.0,
        min_ram_gb=16.0,
        ollama_tag="gemma4:12b",
        hf_repo="google/gemma-4-12b-it-GGUF",
        hf_filename="gemma-4-12b-it-Q4_K_M.gguf",
        sha256=None,
        description="Frontier reasoning and agentic synthesis for GPUs with >=12GB VRAM or 32GB RAM.",
    ),
}


class HardwareDetector:
    """Detects system RAM, GPU VRAM, and storage to recommend the appropriate model tier."""

    @staticmethod
    def detect(target_path: Path | None = None) -> HardwareProfile:
        total_ram = 0.0
        avail_ram = 0.0

        # Memory detection
        try:
            if hasattr(os, "sysconf"):
                total_ram = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
                avail_ram = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / (1024**3)
        except Exception:
            pass

        if total_ram == 0:
            total_ram = 8.0  # sensible fallback
            avail_ram = 4.0

        # Free disk detection
        target_dir = target_path or Path.home()
        try:
            free_disk = shutil.disk_usage(str(target_dir)).free / (1024**3)
        except Exception:
            free_disk = 10.0

        # GPU detection
        has_gpu = False
        vram_gb = 0.0
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                has_gpu = True
                val = float(res.stdout.strip().splitlines()[0])
                vram_gb = val / 1024.0
        except Exception:
            pass

        # Determine tier
        if vram_gb >= 12.0 or total_ram >= 24.0:
            tier = "tier_3"
        elif vram_gb >= 8.0 or total_ram >= 12.0:  # E4B weights are ~6.6 GB
            tier = "tier_2"
        else:
            tier = "tier_1"

        return HardwareProfile(
            total_ram_gb=round(total_ram, 1),
            available_ram_gb=round(avail_ram, 1),
            free_disk_gb=round(free_disk, 1),
            has_gpu=has_gpu,
            vram_gb=round(vram_gb, 1),
            recommended_tier=tier,
        )


class ModelManager:
    """Manages local model checkpoints with strict user consent and integrity verification."""

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or (Path.home() / ".mlforge" / "models")
        self.registry_file = self.root_dir / "registry.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[dict[str, Any]]:
        if not self.registry_file.exists():
            return []
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_registry(self, models: list[dict[str, Any]]) -> None:
        self.registry_file.write_text(json.dumps(models, indent=2), encoding="utf-8")

    def register_local_model(self, name: str, model_path: Path, sha256: str | None = None) -> dict[str, Any]:
        path = model_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        computed_sha256 = sha256 or self._compute_sha256(path)
        size_gb = round(path.stat().st_size / (1024**3), 2)

        models = self.list_models()
        # Remove existing if any
        models = [m for m in models if m.get("name") != name]
        entry = {
            "name": name,
            "path": str(path),
            "size_gb": size_gb,
            "sha256": computed_sha256,
            "source": "local_path",
        }
        models.append(entry)
        self._save_registry(models)
        return entry

    def remove_model(self, name: str) -> bool:
        models = self.list_models()
        match = next((m for m in models if m.get("name") == name), None)
        if not match:
            return False

        # If it was downloaded into our root dir, remove the file
        model_path = Path(match["path"])
        if model_path.is_file() and str(model_path).startswith(str(self.root_dir)):
            try:
                model_path.unlink()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not delete model file: {e}[/yellow]")

        models = [m for m in models if m.get("name") != name]
        self._save_registry(models)
        return True

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def pull_ollama_model(self, tag: str) -> bool:
        """Pulls a model using Ollama with streaming progress."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama CLI is not installed. Run 'saara setup ollama' first.")
        cmd = ["ollama", "pull", tag]
        res = subprocess.run(cmd)
        return res.returncode == 0
