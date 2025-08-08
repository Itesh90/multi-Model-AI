from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi import BackgroundTasks
from .auth import get_current_user, create_api_key
from typing import List, Optional
import os
import uuid
from datetime import datetime
from .services.text_processing import TextProcessor
from .services.image_processing import ImageProcessor
from .services.audio_processing import AudioProcessor
from .services.video_processing import VideoProcessor
# Import our services
from .database import get_db, init_db
from .vector_db import VectorDB
from .storage import FilebaseStorage


# Initialize text processor
text_processor = TextProcessor()
image_processor = ImageProcessor()
audio_processor = AudioProcessor()
video_processor = VideoProcessor() 



app = FastAPI(
    title="Multi-Modal AI Platform (Student Edition)",
    description="A simplified multi-modal AI platform for learning purposes",
    version="0.1.0"
)

# Initialize services
vector_db = VectorDB()
storage = FilebaseStorage(bucket_name="multimodal-student-bucket")  # Must be globally unique!

# API key configuration
API_KEY = "student-api-key-123"  # In real app, use environment variables!
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_current_user(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return {"user_id": 1}  # In real app, this would come from your auth system

@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ Database initialized at startup")
    print("✅ API server is ready to handle requests")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Multi-Modal AI Platform",
        "status": "active",
        "version": "0.1.0",
        "endpoints": [
            "/docs - API documentation",
            "/process-image - Upload and process an image"
        ]
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/user/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "api_key_preview": current_user["api_key"][:8] + "..." if "api_key" in current_user else "N/A"
    }

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process an image with AI
    - Store the image in storage
    - Generate embedding and description
    - Store in vector database
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    # Create safe filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please use JPG or PNG."
        )
    
    safe_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Save to temporary location
    temp_path = f"/tmp/{safe_filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read the image data for processing
        with open(temp_path, "rb") as img_file:
            image_data = img_file.read()
        
        # Generate embedding
        embedding = image_processor.generate_embedding(image_data)
        
        # Generate description
        description = image_processor.describe_image(image_data)
        
        # Upload to storage
        storage_path = f"images/{current_user['user_id']}/{safe_filename}"
        storage.upload_file(temp_path, storage_path)
        
        # Store in vector database
        vector_db.add_image_description(
            description=description,
            image_path=storage_path,
            user_id=current_user["user_id"],
            embedding=embedding
        )
        
        # Generate a signed URL for viewing
        image_url = storage.generate_signed_url(storage_path)
        
        return {
            "filename": file.filename,
            "storage_path": storage_path,
            "description": description,
            "embedding_dimension": len(embedding),
            "image_url": image_url,
            "user_id": current_user["user_id"]
        }
    
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/text/embedding")
def get_text_embedding(
    text: str,
    current_user: dict = Depends(get_current_user)
):
    """Generate embedding for text"""
    embedding = text_processor.generate_embedding(text)
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "embedding_dimension": len(embedding),
        "embedding": embedding[:5] + ["..."]  # Show first 5 values for brevity
    }

@app.post("/text/sentiment")
def analyze_sentiment(
    text: str,
    current_user: dict = Depends(get_current_user)
):
    """Analyze sentiment of text"""
    return text_processor.analyze_sentiment(text)

@app.post("/text/summarize")
def summarize_text(
    text: str,
    max_length: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Summarize text"""
    return {
        "original_text_length": len(text),
        "summary": text_processor.summarize_text(text, max_length)
    }


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload and process audio
    - Store the audio in storage
    - Generate transcription
    - Store in vector database
    """
    # Check if audio processor is available
    if audio_processor is None:
        raise HTTPException(
            status_code=500,
            detail="Audio processing service is not available"
        )
    
    # Validate file type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="File must be audio (WAV, MP3, etc.)"
        )
    
    # Create safe filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.wav', '.mp3', '.m4a']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please use WAV, MP3, or M4A."
        )
    
    safe_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Save to temporary location
    temp_path = f"/tmp/{safe_filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read the audio data for processing
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Transcribe audio
        transcription = audio_processor.transcribe_audio(audio_data)
        
        # Generate embedding
        embedding = audio_processor.generate_audio_embedding(audio_data)
        
        # Upload to storage
        storage_path = f"audio/{current_user['user_id']}/{safe_filename}"
        storage.upload_file(temp_path, storage_path)
        
        # Store in vector database
        vector_db.add_audio_transcription(
            transcription=transcription["text"],
            audio_path=storage_path,
            user_id=current_user["user_id"],
            embedding=embedding,
            language=transcription["language"]
        )
        
        # Generate a signed URL for access
        audio_url = storage.generate_signed_url(storage_path)
        
        return {
            "filename": file.filename,
            "storage_path": storage_path,
            "transcription": transcription["text"],
            "language": transcription["language"],
            "embedding_dimension": len(embedding),
            "audio_url": audio_url,
            "user_id": current_user["user_id"]
        }
    
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search/audio")
async def search_audio(
    query: str,
    limit: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """
    Search for similar audio transcriptions
    """
    # Generate embedding for the query
    if audio_processor is None:
        raise HTTPException(
            status_code=500,
            detail="Audio processing service is not available"
        )
    
    # Create a dummy audio representation for the text query
    # In a real implementation, you'd use a text-to-audio embedding model
    query_embedding = text_processor.generate_embedding(query)
    
    # Search for similar audio
    results = vector_db.search_similar_audio(query_embedding, limit=limit)
    
    return {
        "results": results,
        "query": query[:50] + "..." if len(query) > 50 else query,
        "embedding_dimension": len(query_embedding)
    }


@app.post("/text/process-and-store")
def process_and_store_text(
    text: str,
    meta dict = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Process text and store in vector database
    - Generate embedding
    - Analyze sentiment
    - Store in vector DB with metadata
    """
    # Generate embedding
    embedding = text_processor.generate_embedding(text)
    
    # Analyze sentiment
    sentiment = text_processor.analyze_sentiment(text)
    
    # Prepare metadata
    meta = metadata or {}
    meta.update({
        "user_id": current_user["user_id"],
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"]
    })
    
    # Store in vector database
    # Note: Our VectorDB class needs a slight update for metadata
    # We'll modify it to accept metadata
    success = vector_db.add_text_with_metadata(
        text=text,
        metadata=meta
    )
    
    return {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "embedding_dimension": len(embedding),
        "sentiment": sentiment,
        "stored": success
    }


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload and process video
    - Store the video in storage
    - Extract key frames
    - Process frames with image processor
    - Transcribe audio
    - Store in vector database
    """
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video (MP4, etc.)"
        )
    
    # Create safe filename
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.mp4', '.mov', '.avi']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please use MP4."
        )
    
    safe_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Save to temporary location
    temp_path = f"/tmp/{safe_filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Read the video data for processing
        with open(temp_path, "rb") as video_file:
            video_data = video_file.read()
        
        # Extract frames
        frames = video_processor.extract_frames(video_data, max_frames=5)
        frame_descriptions = []
        
        # Process each frame with image processor
        for i, frame_data in enumerate(frames):
            description = image_processor.describe_image(frame_data)
            embedding = image_processor.generate_embedding(frame_data)
            
            # Store frame in storage
            frame_path = f"video-frames/{current_user['user_id']}/{safe_filename}_frame_{i}.jpg"
            with tempfile.NamedTemporaryFile(suffix=".jpg") as frame_temp:
                frame_temp.write(frame_data)
                frame_temp.flush()
                storage.upload_file(frame_temp.name, frame_path)
            
            # Store in vector database
            vector_db.add_image_description(
                description=description,
                image_path=frame_path,
                user_id=current_user["user_id"],
                embedding=embedding
            )
            
            frame_descriptions.append({
                "frame_number": i,
                "description": description,
                "frame_path": frame_path
            })
        
        # Transcribe audio from video
        try:
            transcription = video_processor.transcribe_audio_from_video(video_data)
        except Exception as e:
            logger.warning(f"Video audio transcription failed: {str(e)}")
            transcription = {
                "text": "Audio transcription not available",
                "language": "unknown",
                "segments": []
            }
        
        # Upload original video to storage
        storage_path = f"videos/{current_user['user_id']}/{safe_filename}"
        storage.upload_file(temp_path, storage_path)
        
        # Generate a signed URL for access
        video_url = storage.generate_signed_url(storage_path)
        
        return {
            "filename": file.filename,
            "storage_path": storage_path,
            "frame_count": len(frames),
            "frame_descriptions": frame_descriptions,
            "transcription": transcription["text"],
            "language": transcription["language"],
            "video_url": video_url,
            "user_id": current_user["user_id"]
        }
    
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
@app.post("/search/images")
async def search_images(
    query: Optional[str] = None,
    file: Optional[UploadFile] = File(None),
    limit: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """
    Search for similar images
    - By text query OR
    - By uploading a reference image
    """
    if not query and not file:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' text or an image 'file' must be provided"
        )
    
    # Case 1: Text query
    if query and not file:
        results = vector_db.search_similar(query, limit=limit)
        return {"results": results, "query_type": "text"}
    
    # Case 2: Image search
    if file:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image for image search"
            )
        
        # Process the image
        temp_path = f"/tmp/{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        try:
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            
            with open(temp_path, "rb") as img_file:
                image_data = img_file.read()
            
            # Generate embedding for the query image
            query_embedding = image_processor.generate_embedding(image_data)
            
            # Search for similar images
            results = vector_db.search_similar_by_vector(
                query_embedding, 
                limit=limit
            )
            
            return {
                "results": results, 
                "query_type": "image",
                "query_embedding_dimension": len(query_embedding)
            }
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Should never get here
    raise HTTPException(status_code=400, detail="Invalid search parameters")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)