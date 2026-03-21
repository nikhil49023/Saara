"""
SAARA Exception Classes

Custom exceptions for SAARA AI package.
"""


class SaaraException(Exception):
    """Base exception for all SAARA errors."""

    pass


class ConfigurationError(SaaraException):
    """Raised when configuration is invalid."""

    pass


class ModelNotFoundError(SaaraException):
    """Raised when a model cannot be found."""

    pass


class OllamaConnectionError(SaaraException):
    """Raised when Ollama service cannot be reached."""

    pass


class OllamaModelNotFoundError(SaaraException):
    """Raised when a model is not available in Ollama."""

    pass


class TrainingError(SaaraException):
    """Raised when training fails."""

    pass


class EvaluationError(SaaraException):
    """Raised when evaluation fails."""

    pass


class DeploymentError(SaaraException):
    """Raised when deployment fails."""

    pass


class PDFExtractionError(SaaraException):
    """Raised when PDF extraction fails."""

    pass


class ChunkingError(SaaraException):
    """Raised when text chunking fails."""

    pass


class CleaningError(SaaraException):
    """Raised when text cleaning fails."""

    pass


class DatasetGenerationError(SaaraException):
    """Raised when dataset generation fails."""

    pass


class DatasetFormatError(SaaraException):
    """Raised when dataset format is invalid."""

    pass


class LabelingError(SaaraException):
    """Raised when data labeling fails."""

    pass


class RAGError(SaaraException):
    """Raised when RAG operations fail."""

    pass


class APIError(SaaraException):
    """Raised when API operations fail."""

    pass


class DependencyError(SaaraException):
    """Raised when a required dependency is not installed."""

    pass
