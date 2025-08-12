import pytest
from unittest import mock
from backend.vector_db import VectorDB

@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client"""
    with mock.patch('backend.vector_db.weaviate.Client') as mock_client:
        # Configure the mock
        mock_client.return_value = mock.Mock(
            schema=mock.Mock(
                exists=mock.Mock(return_value=False),
                create_class=mock.Mock()
            ),
            data_object=mock.Mock(
                create=mock.Mock(return_value="object_id")
            ),
            query=mock.Mock(
                get=mock.Mock(
                    return_value=mock.Mock(
                        with_near_vector=mock.Mock(return_value=mock.Mock()),
                        with_limit=mock.Mock(return_value=mock.Mock()),
                        do=mock.Mock(return_value={
                            "data": {
                                "Get": {
                                    "ImageDescription": [
                                        {"description": "test", "image_path": "path", "user_id": 1}
                                    ]
                                }
                            }
                        })
                    )
                )
            )
        )
        yield mock_client

def test_vector_db_initialization(mock_weaviate_client):
    """Test VectorDB initialization"""
    db = VectorDB()
    
    # Check if schema methods were called
    assert mock_weaviate_client.return_value.schema.exists.call_count == 4
    assert mock_weaviate_client.return_value.schema.create_class.call_count == 4

def test_store_image_description(mock_weaviate_client):
    """Test storing image description"""
    db = VectorDB()
    result = db.store_image_description(
        description="Test image",
        image_path="test.jpg",
        user_id=1,
        embedding=[0.1] * 512
    )
    
    assert result is True
    # Check if data_object.create was called
    assert mock_weaviate_client.return_value.data_object.create.call_count == 1

def test_search_similar(mock_weaviate_client):
    """Test search functionality"""
    db = VectorDB()
    results = db.search_similar("test query")
    
    # Check results structure
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["description"] == "test"
    
    # Check query methods were called
    query_mock = mock_weaviate_client.return_value.query.get
    assert query_mock.call_count == 1
    assert query_mock.return_value.with_near_text.call_count == 1
    assert query_mock.return_value.with_limit.call_count == 1
    assert query_mock.return_value.do.call_count == 1