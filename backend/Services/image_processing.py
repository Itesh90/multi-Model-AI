
import io
import os
import tempfile
from typing import List, Dict, Any, Optional
import logging

# Third-party libs (may be optional in lightweight student environments)
try:
    import numpy as np
    from PIL import Image
    import cv2
    import torch
    from transformers import CLIPProcessor, CLIPModel
    _HAS_TORCH = True
except Exception as e:  # pragma: no cover - environment dependent
    # Minimal safe fallbacks so the module can still be imported in CI/dev
    _HAS_TORCH = False
    # create tiny stubs for types we reference
    class Image:  # type: ignore
        pass

    np = None
    cv2 = None
    torch = None
    CLIPProcessor = None
    CLIPModel = None

# Configure logger
logger = logging.getLogger("image_processor")
logging.basicConfig(level=logging.INFO)


class ImageProcessor:
    """Handles image preprocessing, embedding generation and simple analysis.

    Notes:
        - Uses CLIP (openai/clip-vit-base-patch32) when available.
        - Falls back to lightweight stubs when heavy libraries are missing (so unit tests
          and static analysis can import the module).
    """

    # Use class-level singletons to avoid re-loading heavy models repeatedly
    _model_instance: Optional[Any] = None
    _processor_instance: Optional[Any] = None

    def __init__(self, device: Optional[str] = None, max_size: int = 500):
        self.max_size = max_size

        if not _HAS_TORCH:
            logger.warning("Torch/Transformers not available. ImageProcessor will operate in limited mode.")
            self.model = None
            self.processor = None
            self.device = None
            return

        # Determine device (force CPU for student-friendly default)
        if device:
            self.device = torch.device(device)
        else:
            # default to CPU; if CUDA is available you can pass device="cuda"
            self.device = torch.device("cpu")

        # Load CLIP model/processor only once
        if not ImageProcessor._model_instance or not ImageProcessor._processor_instance:
            try:
                logger.info("Loading CLIP model and processor (this may take a while)...")
                ImageProcessor._model_instance = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                ImageProcessor._processor_instance = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                logger.exception("Failed to load CLIP model/processor: %s", e)
                # Keep None so methods gracefully degrade
                ImageProcessor._model_instance = None
                ImageProcessor._processor_instance = None

        self.model = ImageProcessor._model_instance
        self.processor = ImageProcessor._processor_instance

        if self.model:
            self.model.eval()
            try:
                self.model.to(self.device)
            except Exception:
                logger.debug("Unable to move model to device; continuing on default device")

        logger.info("ImageProcessor ready (device=%s, max_size=%d)", getattr(self.device, 'type', None), self.max_size)

    # --------------------- Utilities ---------------------
    def _load_pil_image(self, image_data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            return image
        except Exception as e:
            logger.exception("Failed to load image: %s", e)
            raise ValueError("Invalid image data")

    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """Convert bytes to a PIL Image, normalize modes and resize while keeping aspect ratio."""
        image = self._load_pil_image(image_data)

        # Convert RGBA to RGB by compositing on white background
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

        # Resize if larger than max_size
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        logger.debug("Preprocessed image size: %s", image.size)
        return image

    # --------------------- Embeddings ---------------------
    def generate_embedding(self, image_data: bytes) -> List[float]:
        """Generate image embedding using CLIP. Returns a Python list of floats.

        Raises:
            RuntimeError: if model/processor are unavailable
        """
        if not self.model or not self.processor:
            raise RuntimeError("CLIP model/processor not loaded. Cannot generate embeddings.")

        image = self.preprocess_image(image_data)

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            # move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            emb = features[0].cpu().numpy().tolist()

            # Explicitly free GPU memory if used
            if self.device and getattr(self.device, "type", "cpu") != "cpu":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            logger.info("Generated embedding of length %d", len(emb))
            return emb

        except Exception as e:
            logger.exception("Embedding generation failed: %s", e)
            raise

    # --------------------- Simple object detection placeholder ---------------------
    def detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Return a placeholder list of detected objects. Use a proper detector for production."""
        try:
            # Attempt a very lightweight heuristic: edge density -> "object" detection demo
            image = self.preprocess_image(image_data)

            if cv2 is None or np is None:
                logger.debug("OpenCV/NumPy not available; returning static placeholder detections")
                return [
                    {"object": "person", "confidence": 0.9, "bbox": [100, 50, 200, 300], "description": "person"}
                ]

            img_cv = np.array(image)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            edge_density = edges.mean()

            logger.debug("Edge density: %s", edge_density)

            if edge_density > 0.02:  # arbitrary threshold for demo purposes
                return [
                    {"object": "complex_scene", "confidence": 0.7, "bbox": [0, 0, image.size[0], image.size[1]], "description": "Complex scene with multiple items"}
                ]

            return []

        except Exception as e:
            logger.exception("Object detection error: %s", e)
            return []

    # --------------------- Image description (zero-shot via CLIP) ---------------------
    def describe_image(self, image_data: bytes, categories: Optional[List[str]] = None) -> str:
        """Return a textual description using CLIP zero-shot classification.

        If CLIP is unavailable, returns a short generic description.
        """
        if categories is None:
            categories = [
                "a photo of a person", "a photo of a dog", "a photo of a cat",
                "a photo of a car", "a photo of a building", "a photo of nature",
                "a photo of food", "a photo of an outdoor scene", "a photo of an indoor scene"
            ]

        if not self.model or not self.processor:
            logger.debug("CLIP unavailable; returning generic description")
            return "This is an image."

        image = self.preprocess_image(image_data)

        try:
            inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # logits_per_image shape: (batch_size=1, num_texts)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            max_idx = torch.argmax(probs, dim=1).item()

            category = categories[max_idx]
            # Remove leading "a photo of " if present
            description = category.replace("a photo of ", "")
            return f"This image shows {description}."

        except Exception as e:
            logger.exception("Image description failed: %s", e)
            return "This is an image."


# --------------------- Simple CLI / test harness ---------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", "-i", help="Path to test image (JPEG/PNG)")
    args = parser.parse_args()

    if not args.test_image or not os.path.exists(args.test_image):
        # create a simple test image if none provided
        from PIL import ImageDraw
        test_path = "test_generated.jpg"
        img = Image.new("RGB", (400, 300), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test Image", fill=(255, 255, 0))
        img.save(test_path)
        args.test_image = test_path
        print(f"No image provided. Created sample image at {test_path}")

    with open(args.test_image, "rb") as f:
        data = f.read()

    ip = ImageProcessor()

    try:
        print("\nGenerating embedding...")
        emb = None
        try:
            emb = ip.generate_embedding(data)
            print(f"Embedding length: {len(emb)}")
        except Exception as e:
            print(f"Embedding generation skipped/failed: {e}")

        print("\nDetecting objects...")
        objs = ip.detect_objects(data)
        print(objs)

        print("\nDescribing image...")
        desc = ip.describe_image(data)
        print(desc)

    except Exception as e:
        print("Test failed:", e)
