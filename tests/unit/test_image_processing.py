import pytest
from PIL import Image
import io
import numpy as np
from backend.Services.Rag.rag_services import ImageProcessor


@pytest.fixture
def image_processor():
    """Fixture for image processor"""
    return ImageProcessor()

@pytest.fixture
def sample_image_bytes():
    """Fixture for sample image bytes"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_image_preprocessing(image_processor, sample_image_bytes):
    """Test image preprocessing"""
    image = image_processor.preprocess_image(sample_image_bytes)
    
    # Should be a PIL Image
    assert isinstance(image, Image.Image)
    # Should be RGB
    assert image.mode == 'RGB'
    # Should be resized appropriately
    assert max(image.size) <= 500

def test_embedding_generation(image_processor, sample_image_bytes):
    """Test image embedding generation"""
    embedding = image_processor.generate_embedding(sample_image_bytes)
    
    # CLIP uses 512 dimensions
    assert len(embedding) == 512
    # Embeddings should be normalized (approximately)
    assert abs(sum(x*x for x in embedding) - 1.0) < 0.1

def test_image_description(image_processor, sample_image_bytes):
    """Test image description generation"""
    description = image_processor.describe_image(sample_image_bytes)
    
    # Should return a non-empty string
    assert isinstance(description, str)
    assert len(description) > 10
    # Should contain the word "image"
    assert "image" in description.lower()

def test_object_detection(image_processor, sample_image_bytes):
    """Test object detection (placeholder)"""
    objects = image_processor.detect_objects(sample_image_bytes)
    
    # Should return a list
    assert isinstance(objects, list)
    # In our placeholder implementation, should return 2 objects
    assert len(objects) == 2
    # Each object should have required fields
    for obj in objects:
        assert "object" in obj
        assert "confidence" in obj
        assert "bbox" in obj
        assert "description" in obj