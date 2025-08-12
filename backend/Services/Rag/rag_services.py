import logging
import time
from typing import Optional, Dict, Any
from .config import RAGConfig
from .exceptions import RAGServiceError, ConfigurationError
from .models import RAGResponse, SourceDocument, MultiModalRequest
from .retriever import DocumentRetriever
from .generator import ResponseGenerator
from .multimodel_processor import MultiModalProcessor

logger = logging.getLogger(__name__)

class RAGService:
    """Main RAG service orchestrating retrieval and generation"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG service with configuration"""
        logger.info("Initializing RAG service...")
        
        self.config = config or RAGConfig()
        try:
            self.config.validate()
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
        
        self.retriever = None
        self.generator = None
        self.multimodal_processor = MultiModalProcessor()
        
        self._initialize_components()
        logger.info("✅ RAG service initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all service components"""
        try:
            # Initialize retriever
            self.retriever = DocumentRetriever(self.config)
            
            # Initialize generator
            self.generator = ResponseGenerator(self.config, self.retriever.retriever)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RAGServiceError(f"Component initialization failed: {e}")
    
    def generate_response(self, query: str, user_id: Optional[int] = None) -> RAGResponse:
        """Generate response for a query"""
        start_time = time.time()
        
        try:
            logger.info(f"Generating response for query: {query[:50]}...")
            
            # Generate response
            result = self.generator.generate(query)
            
            # Process source documents
            sources = []
            for doc in result.get("source_documents", []):
                sources.append(SourceDocument(
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=doc.metadata.get("_additional", {}).get("score", 0.0)
                ))
            
            processing_time = time.time() - start_time
            
            return RAGResponse(
                query=query,
                result=result["result"],
                sources=sources,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return RAGResponse(
                query=query,
                result=f"I encountered an error processing your request: {str(e)}",
                sources=[],
                processing_time=time.time() - start_time
            )
    
    def generate_multimodal_response(self, request: MultiModalRequest) -> RAGResponse:
        """Generate response for multi-modal request"""
        try:
            # Generate base response
            base_response = self.generate_response(request.query, request.user_id)
            
            # Enhance with multi-modal context
            enhanced_response = self.multimodal_processor.enhance_response(base_response, request)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Multi-modal response generation failed: {e}")
            return RAGResponse(
                query=request.query,
                result=f"I encountered an error processing your multi-modal request: {str(e)}",
                sources=[]
            )