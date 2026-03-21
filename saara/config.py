"""
SAARA Configuration Classes

Provides dataclass-based configuration for all major components with backward
compatibility for dict-based configs.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for LLMTrainer."""

    model_id: str = "sarvamai/sarvam-1"
    output_dir: str = "./models"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    resume_from_checkpoint: Optional[str] = None
    teacher_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PipelineConfig:
    """Configuration for DataPipeline."""

    output_directory: str = "./output"
    model: str = "granite"
    use_ocr: bool = True
    ocr_model: str = "qwen"  # "qwen" or "moondream"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    labeler: Optional[str] = None
    generate_synthetic: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluatorConfig:
    """Configuration for ModelEvaluator."""

    teacher_provider: str = "ollama"  # "ollama", "openai", "deepseek", "gemini", "sarvam"
    teacher_model: str = "granite"
    num_samples: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    allow_improvement_data: bool = True
    improvement_data_threshold: float = 0.6

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EvaluatorConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DeployerConfig:
    """Configuration for ModelDeployer."""

    deployment_type: str = "local"  # "local", "cloud", "huggingface"
    export_format: str = "gguf"  # "gguf", "safetensors"
    quantization: Optional[str] = None  # None, "q4", "q8"
    cloud_provider: Optional[str] = None  # "gcp", "aws", "huggingface"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DeployerConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RAGConfig:
    """Configuration for RAGEngine."""

    vector_store: str = "chromadb"  # "chromadb", "pinecone"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.5
    hybrid_search: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PretrainConfig:
    """Configuration for PretrainingModule."""

    model_name: str = "custom-llm"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 3
    output_dir: str = "./pretrained_models"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PretrainConfig":
        """Create config from dictionary (backward compatibility)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                kwargs[field_name] = config[field_name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Config helper to convert dict to appropriate dataclass
def convert_config(config: Any, config_class: type) -> Any:
    """
    Convert a config dict to the appropriate dataclass, or return as-is if already correct type.

    Args:
        config: Dictionary or dataclass instance
        config_class: Target dataclass type

    Returns:
        Instance of config_class

    Example:
        config = convert_config({"model_id": "..."}, TrainConfig)
    """
    if config is None:
        return config_class()
    elif isinstance(config, dict):
        return config_class.from_dict(config)
    elif isinstance(config, config_class):
        return config
    else:
        raise TypeError(f"Expected dict or {config_class.__name__}, got {type(config)}")
