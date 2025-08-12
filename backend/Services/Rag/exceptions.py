class RAGServiceError(Exception):
    """Base exception for RAG service errors"""
    pass

class ConfigurationError(RAGServiceError):
    """Configuration related errors"""
    pass

class RetrievalError(RAGServiceError):
    """Retrieval related errors"""
    pass

class GenerationError(RAGServiceError):
    """Generation related errors"""
    pass
