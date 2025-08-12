import pytest
from unittest import mock
from backend.Services.Rag.rag_services import RAGService

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("WEAVIATE_URL", "http://mock-weaviate:8080")
    monkeypatch.setenv("WEAVIATE_API_KEY", "mock-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")

@pytest.fixture
def mock_langchain():
    """Mock Langchain components"""
    with mock.patch('backend.services.rag.ChatOpenAI') as mock_llm, \
         mock.patch('backend.services.rag.Weaviate') as mock_vector_store, \
         mock.patch('backend.services.rag.RetrievalQA') as mock_retrieval_qa:
        
        # Configure mocks
        mock_llm.return_value = mock.Mock()
        mock_vector_store.return_value = mock.Mock(
            as_retriever=mock.Mock(
                return_value=mock.Mock()
            )
        )
        mock_retrieval_qa.from_chain_type.return_value = mock.Mock(
            invoke=mock.Mock(
                return_value={
                    "result": "Test response",
                    "source_documents": [
                        mock.Mock(
                            page_content="Document content",
                            metadata={"source": "test"}
                        )
                    ]
                }
            )
        )
        
        yield {
            "llm": mock_llm,
            "vector_store": mock_vector_store,
            "retrieval_qa": mock_retrieval_qa
        }

def test_rag_initialization(mock_env_vars, mock_langchain):
    """Test RAG service initialization"""
    rag = RAGService()
    assert rag is not None

def test_response_generation(mock_env_vars, mock_langchain):
    """Test response generation"""
    rag = RAGService()
    response = rag.generate_response("What is AI?")
    
    # Check response structure
    assert "query" in response
    assert "result" in response
    assert "sources" in response
    assert response["result"] == "Test response"
    assert len(response["sources"]) == 1

def test_multi_modal_response(mock_env_vars, mock_langchain):
    """Test multi-modal response generation"""
    rag = RAGService()
    response = rag.generate_multi_modal_response(
        query="Describe this image",
        image_data=b"mock_image_data"
    )
    
    # Check response structure
    assert "query" in response
    assert "result" in response
    assert "sources" in response
    assert "[Note: This response considers the provided image context.]" in response["result"]