from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class SourceDocument:
    """Represents a source document with metadata"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0

@dataclass
class RAGResponse:
    """Response from RAG service"""
    query: str
    result: str
    sources: List[SourceDocument]
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "query": self.query,
            "result": self.result,
            "sources": [
                {
                    "content": source.content,
                    "metadata": source.metadata,
                    "relevance_score": source.relevance_score
                }
                for source in self.sources
            ],
            "processing_time": self.processing_time
        }

@dataclass
class MultiModalRequest:
    """Multi-modal request data"""
    query: str
    image_data: Optional[bytes] = None
    audio_data: Optional[bytes] = None
    video_data: Optional[bytes] = None
    user_id: Optional[int] = None
