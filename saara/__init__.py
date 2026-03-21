"""
Saara: Autonomous Document-to-LLM Data Factory SDK.

🪔 ज्ञानस्य सारः - The Essence of Knowledge

Powered by Google Gemini 2.0 Flash & Gemma 2 Models

© 2025-2026 Kilani Sai Nikhil. All Rights Reserved.
"""

__version__ = "1.6.4"
__author__ = "Kilani Sai Nikhil"
__copyright__ = "© 2025-2026 Kilani Sai Nikhil. All Rights Reserved."
__license__ = "Proprietary"

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
]
