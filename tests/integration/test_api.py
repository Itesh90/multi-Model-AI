import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "active"
    assert "timestamp" in data
    assert "metrics" in data

@pytest.mark.asyncio
async def test_text_sentiment(client: AsyncClient, api_key: str):
    """Test text sentiment analysis endpoint"""
    response = await client.post(
        "/text/sentiment",
        json={"text": "I love this product! It's amazing!"},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == "POSITIVE"
    assert data["score"] > 0.9

@pytest.mark.asyncio
async def test_text_embedding(client: AsyncClient, api_key: str):
    """Test text embedding endpoint"""
    response = await client.post(
        "/text/embedding",
        json={"text": "This is a test sentence."},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding_dimension" in data
    assert data["embedding_dimension"] == 512
    assert "embedding" in data
    assert len(data["embedding"]) == 5  # Only first 5 values shown

@pytest.mark.asyncio
async def test_image_processing(client: AsyncClient, api_key: str, tmp_path):
    """Test image processing endpoint"""
    # Create a test image
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, format='JPEG')
    
    # Test the endpoint
    with open(img_path, "rb") as f:
        response = await client.post(
            "/process-image",
            headers={"X-API-Key": api_key},
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "description" in data
    assert "embedding_dimension" in data
    assert data["embedding_dimension"] == 512
    assert "image_url" in data
    assert data["user_id"] == 1  # From our mock user

@pytest.mark.asyncio
async def test_multi_modal_search(client: AsyncClient, api_key: str):
    """Test multi-modal search endpoint"""
    response = await client.post(
        "/search/multi-modal",
        json={"query": "test query"},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "text" in data["results"]
    assert "images" in data["results"]
    assert "audio" in data["results"]
    assert "video" in data["results"]

@pytest.mark.asyncio
async def test_rag_generation(client: AsyncClient, api_key: str):
    """Test RAG generation endpoint"""
    response = await client.post(
        "/rag/generate",
        json={"query": "What is the main topic of the documents?"},
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert isinstance(data["response"], str)
    assert isinstance(data["sources"], list)