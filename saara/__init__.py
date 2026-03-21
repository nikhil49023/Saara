"""
Saara: Autonomous Document-to-LLM Data Factory SDK.

🪔 ज्ञानस्य सारः - The Essence of Knowledge

Powered by local-first inference (vLLM and Ollama)

Released under the MIT License.
"""

__version__ = "1.6.7"
__author__ = "Kilani Sai Nikhil"
__copyright__ = "Copyright (c) 2025-2026 Kilani Sai Nikhil"
__license__ = "MIT"

# Core imports (always available)
from .cleaner import TextCleaner, SemanticChunker
from .chunker import TextChunker
from .config import (
    TrainConfig,
    PipelineConfig,
    EvaluatorConfig,
    DeployerConfig,
    RAGConfig,
    PretrainConfig,
    convert_config,
)
from .exceptions import (
    SaaraException,
    ConfigurationError,
    ModelNotFoundError,
    OllamaConnectionError,
    TrainingError,
    EvaluationError,
    DeploymentError,
    PDFExtractionError,
    DatasetGenerationError,
)

# Lazy imports for optional heavy dependencies
def __getattr__(name):
    """Lazy import for heavy dependencies."""

    # Training module (requires torch)
    if name == "LLMTrainer":
        from .train import LLMTrainer
        return LLMTrainer

    # Evaluator (requires torch)
    if name == "ModelEvaluator":
        from .evaluator import ModelEvaluator
        return ModelEvaluator

    # Deployer (may require torch)
    if name == "ModelDeployer":
        from .deployer import ModelDeployer
        return ModelDeployer

    # Pipeline (requires ollama, pdfplumber)
    if name == "DataPipeline":
        from .pipeline import DataPipeline
        return DataPipeline

    if name == "PipelineResult":
        from .pipeline import PipelineResult
        return PipelineResult

    # Dataset generator
    if name == "DatasetGenerator":
        from .dataset_generator import DatasetGenerator
        return DatasetGenerator

    # Labeler
    if name == "DataLabeler":
        from .labeler import DataLabeler
        return DataLabeler

    # PDF Extractor
    if name == "PDFExtractor":
        from .pdf_extractor import PDFExtractor
        return PDFExtractor

    # Synthetic generator
    if name == "SyntheticDataGenerator":
        from .synthetic_generator import SyntheticDataGenerator
        return SyntheticDataGenerator

    if name == "DataType":
        from .synthetic_generator import DataType
        return DataType

    if name == "QualityJudge":
        from .synthetic_generator import QualityJudge
        return QualityJudge

    # Accelerator
    if name == "NeuralAccelerator":
        from .accelerator import NeuralAccelerator
        return NeuralAccelerator

    if name == "create_accelerator":
        from .accelerator import create_accelerator
        return create_accelerator

    # Visualizer
    if name == "TrainingDashboard":
        from .visualizer import TrainingDashboard
        return TrainingDashboard

    if name == "ModelAnalyzer":
        from .visualizer import ModelAnalyzer
        return ModelAnalyzer

    if name == "create_visualizer":
        from .visualizer import create_visualizer
        return create_visualizer

    # Cloud Runtime
    if name == "CloudRuntime":
        from .cloud_runtime import CloudRuntime
        return CloudRuntime

    if name == "setup_colab":
        from .cloud_runtime import setup_colab
        return setup_colab

    if name == "is_cloud_environment":
        from .cloud_runtime import is_cloud_environment
        return is_cloud_environment

    # AI Tokenizer
    if name == "AIEnhancedTokenizer":
        from .ai_tokenizer import AIEnhancedTokenizer
        return AIEnhancedTokenizer

    if name == "create_ai_tokenizer":
        from .ai_tokenizer import create_ai_tokenizer
        return create_ai_tokenizer

    # Token Storage (Pre-tokenization)
    if name == "TokenStorage":
        from .token_storage import TokenStorage
        return TokenStorage

    if name == "TokenStorageConfig":
        from .token_storage import TokenStorageConfig
        return TokenStorageConfig

    if name == "quick_tokenize":
        from .token_storage import quick_tokenize
        return quick_tokenize

    # Training Pipeline (Modular)
    if name == "TrainingPipeline":
        from .training_pipeline import TrainingPipeline
        return TrainingPipeline

    if name == "TrainingPipelineConfig":
        from .training_pipeline import TrainingPipelineConfig
        return TrainingPipelineConfig

    if name == "quick_train":
        from .training_pipeline import quick_train
        return quick_train

    # RAG Engine
    if name == "RAGEngine":
        from .rag_engine import RAGEngine
        return RAGEngine

    if name == "RAGManager":
        from .rag_engine import RAGManager
        return RAGManager

    if name == "create_rag_engine":
        from .rag_engine import create_rag_engine
        return create_rag_engine

    if name == "quick_rag":
        from .rag_engine import quick_rag
        return quick_rag

    # LLM Providers (unified API)
    if name == "UnifiedLLM":
        from .llm_providers import UnifiedLLM
        return UnifiedLLM

    if name == "create_llm":
        from .llm_providers import create_llm
        return create_llm

    if name == "quick_generate":
        from .llm_providers import quick_generate
        return quick_generate

    # Tokenizers (flexible, built-in BPE/WordPiece)
    if name == "create_tokenizer":
        from .tokenizers import create_tokenizer
        return create_tokenizer

    if name == "TokenizerRegistry":
        from .tokenizers import TokenizerRegistry
        return TokenizerRegistry

    if name == "BPETokenizer":
        from .tokenizers import BPETokenizer
        return BPETokenizer

    if name == "WordPieceTokenizer":
        from .tokenizers import WordPieceTokenizer
        return WordPieceTokenizer

    if name == "ByteTokenizer":
        from .tokenizers import ByteTokenizer
        return ByteTokenizer

    # File utilities (manual file handling)
    if name == "load_from_file":
        from .file_utils import load_from_file
        return load_from_file

    if name == "save_to_file":
        from .file_utils import save_to_file
        return save_to_file

    if name == "load_jsonl":
        from .file_utils import load_jsonl
        return load_jsonl

    if name == "save_jsonl":
        from .file_utils import save_jsonl
        return save_jsonl

    if name == "extract_texts":
        from .file_utils import extract_texts
        return extract_texts

    if name == "split_dataset":
        from .file_utils import split_dataset
        return split_dataset

    # Quickstart (simple common patterns)
    if name == "QuickLLM":
        from .quickstart import QuickLLM
        return QuickLLM

    if name == "QuickTokenizer":
        from .quickstart import QuickTokenizer
        return QuickTokenizer

    if name == "QuickDataset":
        from .quickstart import QuickDataset
        return QuickDataset

    if name == "QuickFineTune":
        from .quickstart import QuickFineTune
        return QuickFineTune

    if name == "ollama_local":
        from .quickstart import ollama_local
        return ollama_local

    if name == "vllm_local":
        from .quickstart import vllm_local
        return vllm_local

    # QuickAPI (dead-simple end-to-end pipeline)
    if name == "quickapi":
        from . import quickapi as _quickapi
        return _quickapi

    raise AttributeError(f"module 'saara' has no attribute '{name}'")


__all__ = [
    # Config classes
    "TrainConfig",
    "PipelineConfig",
    "EvaluatorConfig",
    "DeployerConfig",
    "RAGConfig",
    "PretrainConfig",
    "convert_config",
    # Exception classes
    "SaaraException",
    "ConfigurationError",
    "ModelNotFoundError",
    "OllamaConnectionError",
    "TrainingError",
    "EvaluationError",
    "DeploymentError",
    "PDFExtractionError",
    "DatasetGenerationError",
    # Core Pipeline
    "DataPipeline",
    "PipelineResult",
    "DatasetGenerator",
    "DataLabeler",
    "PDFExtractor",
    "TextChunker",
    "TextCleaner",
    "SemanticChunker",
    "SyntheticDataGenerator",
    "DataType",
    "QualityJudge",
    # Training & Evaluation
    "LLMTrainer",
    "ModelEvaluator",
    "ModelDeployer",
    # Accelerator & Visualizer
    "NeuralAccelerator",
    "create_accelerator",
    "TrainingDashboard",
    "ModelAnalyzer",
    "create_visualizer",
    # Cloud Runtime
    "CloudRuntime",
    "setup_colab",
    "is_cloud_environment",
    # AI Tokenizer
    "AIEnhancedTokenizer",
    "create_ai_tokenizer",
    # Token Storage & Modular Training
    "TokenStorage",
    "TokenStorageConfig",
    "quick_tokenize",
    "TrainingPipeline",
    "TrainingPipelineConfig",
    "quick_train",
    # RAG Engine
    "RAGEngine",
    "RAGManager",
    "create_rag_engine",
    "quick_rag",
    # LLM Providers (unified API)
    "UnifiedLLM",
    "create_llm",
    "quick_generate",
    # Tokenizers (flexible)
    "create_tokenizer",
    "TokenizerRegistry",
    "BPETokenizer",
    "WordPieceTokenizer",
    "ByteTokenizer",
    # File utilities
    "load_from_file",
    "save_to_file",
    "load_jsonl",
    "save_jsonl",
    "extract_texts",
    "split_dataset",
    # Quickstart
    "QuickLLM",
    "QuickTokenizer",
    "QuickDataset",
    "QuickFineTune",
    "ollama_local",
    "vllm_local",
]
