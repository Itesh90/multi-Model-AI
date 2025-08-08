import io
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles all image processing tasks for the multi-modal platform"""
    
    def __init__(self):
        """Initialize image processing models"""
        logger.info("Initializing image processing models...")
        
        try:
            # Load CLIP model for image embeddings (CPU version for students)
            logger.info("Loading CLIP model for embeddings...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP model loaded successfully")
            
            # Set to evaluation mode (important for inference)
            self.model.eval()
            
            # Move to CPU (students rarely have CUDA-capable GPUs)
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize image processor: {str(e)}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image data for model input
        
        Args:
            image_data: Raw image bytes from upload
            
        Returns:
            PIL Image ready for processing
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if it's RGBA (like PNG with transparency)
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            # Resize while maintaining aspect ratio
            max_size = 500  # Students often have limited resources
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple([int(x * ratio) for x in image.size])
                image = image.resize(new_size, Image.LANCZOS)
            
            logger.debug(f"Preprocessed image size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")
    
    def generate_embedding(self, image_data: bytes) -> List[float]:
        """
        Generate embedding vector for an image
        
        Args:
            image_data: Raw image bytes from upload
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            # Preprocess the image
            image = self.preprocess_image(image_data)
            
            # Process with CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Convert to numpy and then to Python list
            embedding = outputs[0].cpu().numpy().tolist()
            
            logger.info(f"Generated embedding of dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]:
        """
        Detect objects in an image (simplified version for students)
        
        Args:
            image_data: Raw image bytes from upload
            
        Returns:
            List of detected objects with their properties
        """
        try:
            # For students without GPU, we'll use a simpler approach
            # In a real implementation, you'd use a proper object detection model
            
            # Convert to OpenCV format for basic processing
            image = self.preprocess_image(image_data)
            image_cv = np.array(image)
            # Convert RGB to BGR (OpenCV uses BGR)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # This is a placeholder - in a real system you'd use YOLO or similar
            # For learning purposes, we'll return a simple response
            return [
                {
                    "object": "person",
                    "confidence": 0.92,
                    "bbox": [100, 50, 200, 300],  # [x, y, width, height]
                    "description": "A person standing near a building"
                },
                {
                    "object": "building",
                    "confidence": 0.87,
                    "bbox": [50, 100, 300, 250],
                    "description": "A modern office building with glass windows"
                }
            ]
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            # Return empty list instead of failing completely
            return []
    
    def describe_image(self, image_data: bytes) -> str:
        """
        Generate a textual description of an image
        
        Args:
            image_data: Raw image bytes from upload
            
        Returns:
            Text description of the image
        """
        try:
            # In a real implementation, you'd use a vision-language model
            # For students, we'll use a simpler approach with CLIP
            
            # Common categories to check against
            categories = [
                "a photo of a person", "a photo of a dog", "a photo of a cat",
                "a photo of a car", "a photo of a building", "a photo of nature",
                "a photo of food", "a photo of an outdoor scene", "a photo of an indoor scene"
            ]
            
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            # Process with CLIP for zero-shot classification
            inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the prediction
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            max_idx = torch.argmax(probs, dim=1).item()
            
            # Return the description
            description = categories[max_idx].replace("a photo of ", "")
            logger.info(f"Image described as: {description}")
            return f"This image shows {description}."
            
        except Exception as e:
            logger.error(f"Image description failed: {str(e)}")
            return "This is an image."

# Test the processor
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    processor = ImageProcessor()
    
    # Try to load a test image (create one if needed)
    try:
        with open("test.jpg", "rb") as f:
            image_data = f.read()
        
        print("\nTesting image embedding generation...")
        embedding = processor.generate_embedding(image_data)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        print("\nTesting object detection...")
        objects = processor.detect_objects(image_data)
        for obj in objects:
            print(f"- {obj['object']} ({obj['confidence']:.2f}): {obj['description']}")
        
        print("\nTesting image description...")
        description = processor.describe_image(image_data)
        print(f"Description: {description}")
        
    except FileNotFoundError:
        print("❌ Test failed: Please create a 'test.jpg' file in this directory")
        print("You can use any JPEG image for testing")