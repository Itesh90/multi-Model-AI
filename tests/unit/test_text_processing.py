import pytest
from backend.Services.Rag.rag_services import TextProcessor

@pytest.fixture
def text_processor():
    """Fixture for text processor"""
    return TextProcessor()

def test_sentiment_analysis(text_processor):
    """Test sentiment analysis with positive text"""
    result = text_processor.analyze_sentiment("I love this product! It's amazing!")
    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.9

def test_sentiment_analysis_negative(text_processor):
    """Test sentiment analysis with negative text"""
    result = text_processor.analyze_sentiment("This is terrible. I hate it.")
    assert result["label"] == "NEGATIVE"
    assert result["score"] > 0.8

def test_embedding_generation(text_processor):
    """Test embedding generation"""
    text = "This is a test sentence for embedding generation."
    embedding = text_processor.generate_embedding(text)
    
    # CLIP uses 512 dimensions
    assert len(embedding) == 512
    # Embeddings should be normalized (approximately)
    assert abs(sum(x*x for x in embedding) - 1.0) < 0.1

def test_summarization(text_processor):
    """Test text summarization"""
    long_text = ("This is a longer text that needs to be summarized. " * 10) + \
                "It should be condensed to its most important points."
    
    summary = text_processor.summarize_text(long_text, max_length=50)
    
    # Summary should be shorter than original
    assert len(summary) < len(long_text)
    # Summary should contain key information
    assert "summarized" in summary.lower() or "condensed" in summary.lower()

def test_chunking(text_processor):
    """Test text chunking functionality"""
    long_text = "Sentence one. " * 100  # Create long text
    
    chunks = text_processor.chunk_text(long_text, max_tokens=100)
    
    # Should create multiple chunks
    assert len(chunks) > 1
    # Each chunk should be reasonable size
    for chunk in chunks:
        assert len(chunk.split()) <= 110  # Allow some buffer