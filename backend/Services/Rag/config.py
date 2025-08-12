import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration settings for RAG service with OpenRouter support"""

    # Vector store
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None

    # LLM API keys
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Model and generation settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    retrieval_k: int = 3

    # Embedding/local options
    use_local_embeddings: bool = True

    def __post_init__(self):
        """Load environment variables and configure OpenRouter if present"""
        # Vector store
        self.weaviate_url = self.weaviate_url or os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = self.weaviate_api_key or os.getenv("WEAVIATE_API_KEY")

        # API keys
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        # Embeddings toggle
        self.use_local_embeddings = (
            os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        )

        # Model selection via env: RAG_MODEL may be a key from openrouter_models
        rag_model_key = os.getenv("RAG_MODEL")
        if rag_model_key:
            try:
                from .openrouter_models import get_model_info

                model_info = get_model_info(rag_model_key)
                if model_info and model_info.get("name"):
                    self.model_name = model_info["name"]
            except Exception:
                # Fallback silently if helper not available
                pass

        # If OpenRouter is configured, set OpenAI-compatible env so LangChain uses it
        if self.openrouter_api_key:
            # Configure OpenAI client to route via OpenRouter
            os.environ["OPENAI_API_KEY"] = self.openrouter_api_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            # Prefer an OpenRouter-free/default model if not explicitly set
            if not rag_model_key and self.model_name == "gpt-3.5-turbo":
                # Reasonable free default per OpenRouter
                self.model_name = "meta-llama/llama-3.1-8b-instruct:free"

    def validate(self) -> None:
        """Validate required configuration"""
        if not self.weaviate_url:
            raise ValueError("WEAVIATE_URL is required")

        # Accept either OpenAI or OpenRouter configuration
        if not (self.openai_api_key or self.openrouter_api_key or os.getenv("OPENAI_API_KEY")):
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY is required")
