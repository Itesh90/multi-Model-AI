from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
import tempfile
import uuid
import os
import logging
import psutil
import secrets
import re

# -------------------- Logger --------------------
logger = logging.getLogger("multimodal_api")
logging.basicConfig(level=logging.INFO)

# -------------------- Safe optional imports / stubs --------------------
# Try to import your real modules; otherwise provide minimal stubs to keep the app importable.
try:
    from .auth import get_current_user as real_get_current_user, create_api_key, auth
except Exception:
    real_get_current_user = None
    create_api_key = None
    auth = None

try:
    from .Services.text_processing import TextProcessor
    from .Services.image_processing import ImageProcessor
    from .Services.audio_processing import AudioProcessor
    from .Services.video_processing import VideoProcessor
except Exception:
    # Provide simple stubs
    class TextProcessor:
        def generate_embedding(self, text):
            return [0.0]
        def analyze_sentiment(self, text):
            return {"label": "neutral", "score": 0.0}
        def summarize_text(self, text, max_length=100):
            return text[:max_length]

    class ImageProcessor:
        def generate_embedding(self, data):
            return [0.0]
        def describe_image(self, data):
            return "(no description - image processor not configured)"

    class AudioProcessor:
        def transcribe_audio(self, data):
            return {"text": "(transcription unavailable)", "language": "unknown"}
        def generate_audio_embedding(self, data):
            return [0.0]

    class VideoProcessor:
        def extract_frames(self, data, max_frames=5):
            return []
        def transcribe_audio_from_video(self, data):
            return {"text": "(transcription unavailable)", "language": "unknown", "segments": []}

try:
    from .vector_db import VectorDB
except Exception:
    class VectorDB:
        def __init__(self):
            self._store = []
        def add_image_description(self, **kwargs):
            self._store.append({"type": "image", **kwargs})
            return True
        def add_audio_transcription(self, **kwargs):
            self._store.append({"type": "audio", **kwargs})
            return True
        def add_text_with_metadata(self, **kwargs):
            self._store.append({"type": "text", **kwargs})
            return True
        def search_similar(self, query, limit=5):
            return []
        def search_similar_by_vector(self, vec, limit=5):
            return []
        def search_across_modalities(self, query, limit=5, filters=None):
            return {"text": [], "images": [], "audio": [], "video": []}
        def search_text_content_by_vector(self, query_embedding, limit=5, filters=None):
            return []

try:
    from .storage import FilebaseStorage
except Exception:
    class FilebaseStorage:
        def __init__(self, bucket_name: str = "dev-bucket"):
            self.bucket_name = bucket_name
        def upload_file(self, local_path, dest_path):
            logger.info(f"(stub) upload {local_path} -> {dest_path}")
            return True
        def generate_signed_url(self, path, expires=3600):
            return f"https://storage.example/{self.bucket_name}/{path}"

try:
    from .Services.Rag.rag_services import RAGService
except Exception:
    class RAGService:
        def generate_response(self, query, user_id=None):
            return {"result": "(RAG unavailable)", "sources": []}
        def generate_multi_modal_response(self, query, image_data=None, user_id=None):
            return {"result": "(RAG unavailable)", "sources": []}

# Minimal content moderator stub
class ContentModerator:
    def moderate_text(self, text):
        return {"is_safe": True, "censored_text": text, "categories": []}
    def moderate_image(self, description):
        return {"is_safe": True, "categories": []}

content_moderator = ContentModerator()

# Monitoring, rate limiter, and security middleware stubs (no-ops)
try:
    from .middleware.rate_limiter import rate_limit_middleware, rate_limiter
except Exception:
    async def rate_limit_middleware(request, call_next):
        return await call_next(request)
    class _RateLimiter:
        def __init__(self):
            self.request_counts = {}
            self.requests_per_minute = 60
            self.window_seconds = 60
    rate_limiter = _RateLimiter()

try:
    from .middleware.monitoring import monitoring_middleware, monitor
except Exception:
    async def monitoring_middleware(request, call_next):
        return await call_next(request)
    class _Monitor:
        def get_metrics(self):
            return {"status": "ok", "total_requests": 0, "uptime_seconds": 1, "error_rate": 0.0, "endpoint_performance": {}}
    monitor = _Monitor()

try:
    from .middleware.security import security_middleware, add_security_headers
except Exception:
    async def security_middleware(request, call_next):
        return await call_next(request)
    def add_security_headers(response):
        return response

# -------------------- Instantiate app and attach middleware --------------------
app = FastAPI(
    title="Multi-Modal AI Platform (Student Edition)",
    description="A simplified multi-modal AI platform for learning purposes",
    version="0.1.0",
)

# CORS - tune for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# attach optional middleware (these are http middlewares taking request and call_next)
try:
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(monitoring_middleware)
    app.middleware("http")(security_middleware)
except Exception as e:
    logger.warning(f"Could not attach one or more middlewares: {e}")

# -------------------- Globals / services --------------------
text_processor = TextProcessor()
image_processor = ImageProcessor()
audio_processor = AudioProcessor()
video_processor = VideoProcessor()
vector_db = VectorDB()
storage = FilebaseStorage(bucket_name=os.getenv("BUCKET_NAME", "multimodal-student-bucket"))
try:
    rag_service = RAGService()
    logger.info("RAG service initialized")
except Exception as e:
    logger.warning(f"RAG service failed to initialize: {e}")
    rag_service = None

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY = os.getenv("STUDENT_API_KEY", "student-api-key-123")

# Provide a default get_current_user dependency if your auth isn't available
async def get_current_user(api_key: str = Depends(api_key_header)) -> Dict[str, Any]:
    if real_get_current_user:
        return await real_get_current_user(api_key)

    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    # Minimal user object for testing
    return {"user_id": 1, "id": 1, "email": "student@example.com", "api_key": API_KEY}

# -------------------- Utility functions --------------------

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return {
        "memory_percent": process.memory_percent(),
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "cpu_percent": process.cpu_percent(interval=0.1),
    }

# -------------------- TaskManager --------------------
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def create_task(self, task_id: str, coro, description: str = ""):
        async with self.lock:
            if task_id in self.tasks:
                raise ValueError("Task id already exists")

            loop = asyncio.get_event_loop()

            async def _runner():
                self.tasks[task_id]["status"] = "running"
                try:
                    result = await coro
                    self.tasks[task_id]["result"] = result
                    self.tasks[task_id]["status"] = "completed"
                    self.tasks[task_id]["completed_at"] = datetime.now(timezone.utc).timestamp()
                except Exception as e:
                    logger.exception("Background task failed")
                    self.tasks[task_id]["status"] = "failed"
                    self.tasks[task_id]["error"] = str(e)

            self.tasks[task_id] = {
                "id": task_id,
                "description": description,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).timestamp(),
                "result": None,
            }

            # schedule the runner
            loop.create_task(_runner())
            return self.tasks[task_id]

    async def get_task_status(self, task_id: str):
        return self.tasks.get(task_id)

    async def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        now_ts = datetime.now(timezone.utc).timestamp()
        async with self.lock:
            to_delete = [tid for tid, t in self.tasks.items() if t.get("completed_at") and (now_ts - t["completed_at"]) > max_age_seconds]
            for tid in to_delete:
                del self.tasks[tid]


task_manager = TaskManager()

# Periodic cleanup
async def _cleanup_loop():
    while True:
        try:
            await task_manager.cleanup_completed_tasks()
        except Exception:
            logger.exception("Error during task cleanup")
        await asyncio.sleep(300)

@app.on_event("startup")
async def _startup():
    # try to initialize database etc. If you have init_db use it here
    try:
        from .database import init_db
        init_db()
        logger.info("Database initialized at startup")
    except Exception:
        logger.info("No init_db available or failed to run (continuing)")

    # start cleanup loop
    asyncio.create_task(_cleanup_loop())
    logger.info("Startup complete")

# -------------------- Pydantic models --------------------
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @validator("text")
    def validate_text(cls, v: str) -> str:
        if re.search(r"<script|javascript:", v, re.IGNORECASE):
            raise ValueError("Invalid characters detected")
        return v

class ImageRequest(BaseModel):
    max_results: int = Field(5, ge=1, le=20)

# -------------------- Endpoints --------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Multi-Modal AI Platform",
        "status": "active",
        "version": "0.1.0",
        "endpoints": ["/docs - API documentation", "/process-image - Upload and process an image"]
    }

@app.get("/health")
def health_check():
    metrics = monitor.get_metrics() if monitor else {"status": "unknown"}
    return {
        "status": "active",
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "metrics": metrics,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/user/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user.get("user_id"),
        "email": current_user.get("email"),
        "api_key_preview": (current_user.get("api_key")[:8] + "...") if current_user.get("api_key") else "N/A"
    }

# -------------------- Image Processing (supports background tasks) --------------------
@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Please use JPG or PNG.")

    task_id = str(uuid.uuid4())

    async def _task():
        # create a temp file and safely process it
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp.close()

            # read back
            with open(tmp.name, "rb") as fh:
                img_bytes = fh.read()

            description = image_processor.describe_image(img_bytes)
            embedding = image_processor.generate_embedding(img_bytes)

            storage_path = f"images/{current_user['user_id']}/{uuid.uuid4()}{file_extension}"
            try:
                storage.upload_file(tmp.name, storage_path)
            except Exception:
                logger.exception("Failed to upload to storage")

            try:
                vector_db.add_image_description(
                    description=description,
                    image_path=storage_path,
                    user_id=current_user["user_id"],
                    embedding=embedding,
                )
            except Exception:
                logger.exception("Failed to add image to vector DB")

            image_url = storage.generate_signed_url(storage_path)

            # moderation (non-blocking) - student edition: just log
            mod = content_moderator.moderate_image(description)
            if not mod.get("is_safe", True):
                logger.warning("Image moderated: %s", mod)

            return {
                "filename": file.filename,
                "storage_path": storage_path,
                "description": description,
                "embedding_dimension": len(embedding) if hasattr(embedding, "__len__") else None,
                "image_url": image_url,
                "user_id": current_user["user_id"],
            }
        finally:
            if tmp and os.path.exists(tmp.name):
                os.remove(tmp.name)

    # schedule task
    await task_manager.create_task(task_id, _task(), f"Process image {file.filename}")

    return {"task_id": task_id, "status": "processing", "message": "Image processing started in background"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, current_user: dict = Depends(get_current_user)):
    status = await task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

# -------------------- Text endpoints --------------------
@app.post("/text/embedding")
def get_text_embedding(text: str, current_user: dict = Depends(get_current_user)):
    embedding = text_processor.generate_embedding(text)
    preview = embedding[:5] if hasattr(embedding, "__len__") else None
    return {"text": text[:50] + "..." if len(text) > 50 else text, "embedding_dimension": len(embedding), "embedding_preview": preview}

@app.post("/text/summarize")
def summarize_text(text: str, max_length: int = 100, current_user: dict = Depends(get_current_user)):
    return {"original_text_length": len(text), "summary": text_processor.summarize_text(text, max_length)}

@app.post("/text/sentiment")
def analyze_sentiment(request: TextRequest, current_user: dict = Depends(get_current_user)):
    # moderation
    mod = content_moderator.moderate_text(request.text)
    if not mod.get("is_safe", True):
        raise HTTPException(status_code=400, detail=f"Content contains prohibited material: {mod.get('categories')}")
    return text_processor.analyze_sentiment(mod.get("censored_text", request.text))

@app.post("/text/process-and-store")
def process_and_store_text(text: str, meta: Optional[Dict[str, Any]] = None, current_user: dict = Depends(get_current_user)):
    embedding = text_processor.generate_embedding(text)
    sentiment = text_processor.analyze_sentiment(text)
    metadata = meta or {}
    metadata.update({"user_id": current_user["user_id"], "sentiment": sentiment.get("label"), "sentiment_score": sentiment.get("score")})

    try:
        success = vector_db.add_text_with_metadata(text=text, metadata=metadata)
    except Exception:
        logger.exception("Failed to store text in vector DB")
        success = False

    return {"text_preview": text[:100] + "..." if len(text) > 100 else text, "embedding_dimension": len(embedding), "sentiment": sentiment, "stored": success}

# -------------------- Search endpoints (examples) --------------------
@app.post("/search/images")
async def search_images(query: Optional[str] = None, file: Optional[UploadFile] = File(None), limit: int = 5, request_model: ImageRequest = Depends(), current_user: dict = Depends(get_current_user)):
    limit = request_model.max_results
    if not query and not file:
        raise HTTPException(status_code=400, detail="Either 'query' or 'file' must be provided")

    if query and not file:
        results = vector_db.search_similar(query, limit=limit)
        return {"results": results, "query_type": "text"}

    if file:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            tmp.write(await file.read())
            tmp.flush()
            tmp.close()
            with open(tmp.name, "rb") as fh:
                image_data = fh.read()
            query_embedding = image_processor.generate_embedding(image_data)
            results = vector_db.search_similar_by_vector(query_embedding, limit=limit)
            return {"results": results, "query_type": "image", "query_embedding_dimension": len(query_embedding)}
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

# -------------------- Multi-modal RAG --------------------
@app.post("/rag/generate")
async def generate_response(query: str, current_user: dict = Depends(get_current_user)):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service is not available")
    try:
        response = rag_service.generate_response(query=query, user_id=current_user["user_id"])
        return {"query": query, "response": response.get("result"), "sources": response.get("sources"), "user_id": current_user["user_id"]}
    except Exception as e:
        logger.exception("RAG generation failed")
        raise HTTPException(status_code=500, detail=f"Response generation failed: {e}")

@app.post("/rag/multi-modal")
async def multi_modal_rag(query: str, file: Optional[UploadFile] = File(None), current_user: dict = Depends(get_current_user)):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service is not available")
    image_bytes = None
    if file and file.content_type.startswith("image/"):
        image_bytes = await file.read()
    try:
        response = rag_service.generate_multi_modal_response(query=query, image_data=image_bytes, user_id=current_user["user_id"])
        return {"query": query, "response": response.get("result"), "sources": response.get("sources"), "user_id": current_user["user_id"], "has_image_context": image_bytes is not None}
    except Exception as e:
        logger.exception("Multi-modal RAG failed")
        raise HTTPException(status_code=500, detail=f"Response generation failed: {e}")

# -------------------- Monitoring / resources --------------------
@app.get("/system/resources")
async def system_resources(current_user: dict = Depends(get_current_user)):
    return {
        "memory": get_memory_usage(),
        "cache_stats": {"item_count": 0, "memory_estimate_mb": 0},
        "task_stats": {"total_tasks": len(task_manager.tasks), "pending_tasks": sum(1 for t in task_manager.tasks.values() if t["status"] == "pending")},
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    }

# -------------------- Run (dev) --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("multimodal_api_app:app", host="0.0.0.0", port=8000, reload=True)
