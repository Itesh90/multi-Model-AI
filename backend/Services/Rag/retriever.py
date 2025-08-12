import logging
from typing import List, Optional
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from .config import RAGConfig
from .exceptions import RetrievalError
from .models import SourceDocument

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document retrieval from vector store"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.retriever = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize Weaviate vector store"""
        try:
            self.vector_store = Weaviate(
                weaviate_url=self.config.weaviate_url,
                index_name="TextContent",
                text_key="content",
                api_key=self.config.weaviate_api_key,
                attributes=["content_type", "source_type", "source_path", "user_id", "metadata"]
            )
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            logger.info("✅ Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise RetrievalError(f"Vector store initialization failed: {e}")
    
    def retrieve_documents(self, query: str, user_id: Optional[int] = None) -> List[SourceDocument]:
        """Retrieve relevant documents for a query"""
        try:
            if not self.retriever:
                raise RetrievalError("Retriever not initialized")
            
            # Add user context if available
            search_query = f"[USER: {user_id}] {query}" if user_id else query
            
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(search_query)
            
            # Convert to SourceDocument objects
            source_docs = []
            for doc in docs:
                source_docs.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=doc.metadata.get("_additional", {}).get("score", 0.0)
                ))
            
            logger.info(f"Retrieved {len(source_docs)} documents for query")
            return source_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {e}")
