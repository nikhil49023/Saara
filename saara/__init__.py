"""
Saara: Autonomous Document-to-LLM Data Factory SDK.

🪔 ज्ञानस्य सारः - The Essence of Knowledge

© 2024-2025 Kilani Sai Nikhil. All Rights Reserved.
"""

__version__ = "1.2.17"
__author__ = "Kilani Sai Nikhil"
__copyright__ = "© 2024-2025 Kilani Sai Nikhil. All Rights Reserved."
__license__ = "Proprietary"

# Core imports (always available)
from .cleaner import TextCleaner, SemanticChunker
from .chunker import TextChunker

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
    
    raise AttributeError(f"module 'saara' has no attribute '{name}'")


__all__ = [
    "DataPipeline",
    "PipelineResult",
    "LLMTrainer",
    "ModelEvaluator",
    "ModelDeployer",
    "DatasetGenerator",
    "DataLabeler",
    "PDFExtractor",
    "TextChunker",
    "TextCleaner",
    "SemanticChunker", 
    "SyntheticDataGenerator",
    "DataType",
    "QualityJudge",
]
