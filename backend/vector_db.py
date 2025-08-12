import os
import weaviate
from weaviate.auth import AuthApiKey

class VectorDB:
    def __init__(self):
        # In a real app, these would come from environment variables
        self.url = "WEAVIATE_URL"
        self.api_key = "WEAVIATE_API_KEY"
        
        self.client = weaviate.Client(
            url=f"https://{self.url}",
            auth_client_secret=AuthApiKey(api_key=self.api_key),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
            }
        )
        self._setup_schema()
    
    def _setup_schema(self):
        """Create schema if it doesn't exist"""
        try:
            # ImageDescription schema
            if not self.client.schema.exists("ImageDescription"):
                image_class = {
                    "class": "ImageDescription",
                    "description": "Descriptions of images for semantic search",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "description",
                            "dataType": ["text"],
                            "description": "The image description text"
                        },
                        {
                            "name": "image_path",
                            "dataType": ["text"],
                            "description": "Path to the stored image"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["int"],
                            "description": "ID of the user who uploaded"
                        },
                        {
                            "name": "objects",
                            "dataType": ["text[]"],
                            "description": "Detected objects in the image"
                        }
                    ]
                }
                self.client.schema.create_class(image_class)
                print("✅ ImageDescription schema created")

            # TextContent schema
            if not self.client.schema.exists("TextContent"):
                text_class = {
                    "class": "TextContent",
                    "description": "General text content for semantic search",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The main text content"
                        },
                        {
                            "name": "content_type",
                            "dataType": ["text"],
                            "description": "Type of content (document, note, etc.)"
                        },
                        {
                            "name": "source_type",
                            "dataType": ["text"],
                            "description": "Source type (upload, input, etc.)"
                        },
                        {
                            "name": "source_path",
                            "dataType": ["text"],
                            "description": "Path to the source file or identifier"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["int"],
                            "description": "ID of the user who created this content"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata about the text"
                        }
                    ]
                }
                self.client.schema.create_class(text_class)
                print("✅ TextContent schema created")

            # AudioTranscription schema
            if not self.client.schema.exists("AudioTranscription"):
                audio_class = {
                    "class": "AudioTranscription",
                    "description": "Transcriptions of audio content",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "transcription",
                            "dataType": ["text"],
                            "description": "The transcribed text from audio"
                        },
                        {
                            "name": "audio_path",
                            "dataType": ["text"],
                            "description": "Path to the stored audio file"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["int"],
                            "description": "ID of the user who uploaded"
                        },
                        {
                            "name": "language",
                            "dataType": ["text"],
                            "description": "Detected language of the audio"
                        }
                    ]
                }
                self.client.schema.create_class(audio_class)
                print("✅ AudioTranscription schema created")

            # VideoFrame schema
            if not self.client.schema.exists("VideoFrame"):
                video_class = {
                    "class": "VideoFrame",
                    "description": "Key frames extracted from videos",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "description",
                            "dataType": ["text"],
                            "description": "Description of the video frame"
                        },
                        {
                            "name": "frame_path",
                            "dataType": ["text"],
                            "description": "Path to the stored frame image"
                        },
                        {
                            "name": "video_path",
                            "dataType": ["text"],
                            "description": "Path to the original video"
                        },
                        {
                            "name": "frame_number",
                            "dataType": ["int"],
                            "description": "Position of frame in video"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["int"],
                            "description": "ID of the user who uploaded"
                        }
                    ]
                }
                self.client.schema.create_class(video_class)
                print("✅ VideoFrame schema created")
                
            print("✅ Vector database schema verified and ready")
            
        except Exception as e:
            print(f"Error setting up schema: {e}")

    def store_text_content(self, content, content_type, source_type, source_path, user_id, metadata=None):
        """Store general text content with embedding"""
        try:
            # Generate embedding using text processor
            from .Services.text_processing import TextProcessor
            embedding = TextProcessor().generate_embedding(content)
            
            data_object = {
                "content": content,
                "content_type": content_type,
                "source_type": source_type,
                "source_path": source_path,
                "user_id": user_id,
                "metadata": metadata or {}
            }
            
            self.client.data_object.create(
                data_object,
                "TextContent",
                vector=embedding
            )
            return True
        except Exception as e:
            print(f"Error storing text content: {e}")
            return False

    def store_image_description(self, description, image_path, user_id, objects=None):
        """Store image description with embedding"""
        try:
            # Generate embedding using text processor
            from .Services.text_processing import TextProcessor
            embedding = TextProcessor().generate_embedding(description)
            
            data_object = {
                "description": description,
                "image_path": image_path,
                "user_id": user_id,
                "objects": objects or []
            }
            
            self.client.data_object.create(
                data_object,
                "ImageDescription",
                vector=embedding
            )
            return True
        except Exception as e:
            print(f"Error storing image description: {e}")
            return False

    def store_audio_transcription(self, transcription, audio_path, user_id, language):
        """Store audio transcription with embedding"""
        try:
            # Generate embedding using text processor
            from .Services.text_processing import TextProcessor
            embedding = TextProcessor().generate_embedding(transcription)
            
            data_object = {
                "transcription": transcription,
                "audio_path": audio_path,
                "user_id": user_id,
                "language": language
            }
            
            self.client.data_object.create(
                data_object,
                "AudioTranscription",
                vector=embedding
            )
            return True
        except Exception as e:
            print(f"Error storing audio transcription: {e}")
            return False

    def store_video_frame(self, description, frame_path, video_path, frame_number, user_id):
        """Store video frame description with embedding"""
        try:
            # Generate embedding using text processor
            from .Services.text_processing import TextProcessor
            embedding = TextProcessor().generate_embedding(description)
            
            data_object = {
                "description": description,
                "frame_path": frame_path,
                "video_path": video_path,
                "frame_number": frame_number,
                "user_id": user_id
            }
            
            self.client.data_object.create(
                data_object,
                "VideoFrame",
                vector=embedding
            )
            return True
        except Exception as e:
            print(f"Error storing video frame: {e}")
            return False

    def search_text_content(self, query, limit=5, filters=None):
        """Search text content with optimized query"""
        try:
            # Generate query embedding
            from .Services.text_processing import TextProcessor
            query_embedding = TextProcessor().generate_embedding(query)
            
            # Build optimized query
            query_builder = (
                self.client.query
                .get("TextContent", ["content", "content_type", "source_type", "source_path", "user_id"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .with_additional(["distance"])
            )
            
            # Add filters if provided
            if filters and "user_id" in filters:
                query_builder = query_builder.with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": filters["user_id"]
                })
            
            result = query_builder.do()
            
            # Process results to include relevance score
            results = result["data"]["Get"]["TextContent"]
            for item in results:
                if "_additional" in item:
                    item["relevance"] = 1 - item["_additional"]["distance"]
            
            return results
        except Exception as e:
            print(f"Text content search error: {e}")
            return []

    def search_similar_images(self, query, limit=3, user_id=None):
        """Find similar image descriptions"""
        try:
            # Generate query embedding
            from .Services.text_processing import TextProcessor
            query_embedding = TextProcessor().generate_embedding(query)
            
            query_builder = (
                self.client.query
                .get("ImageDescription", ["description", "image_path", "user_id", "objects"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .with_additional(["distance"])
            )
            
            # Add user filter if provided
            if user_id:
                query_builder = query_builder.with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
            
            result = query_builder.do()
            
            results = result["data"]["Get"]["ImageDescription"]
            for item in results:
                if "_additional" in item:
                    item["relevance"] = 1 - item["_additional"]["distance"]
            
            return results
        except Exception as e:
            print(f"Image search error: {e}")
            return []

    def search_similar_audio(self, query, limit=3, user_id=None):
        """Find similar audio transcriptions"""
        try:
            # Generate query embedding
            from .Services.text_processing import TextProcessor
            query_embedding = TextProcessor().generate_embedding(query)
            
            query_builder = (
                self.client.query
                .get("AudioTranscription", ["transcription", "audio_path", "user_id", "language"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .with_additional(["distance"])
            )
            
            # Add user filter if provided
            if user_id:
                query_builder = query_builder.with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
            
            result = query_builder.do()
            
            results = result["data"]["Get"]["AudioTranscription"]
            for item in results:
                if "_additional" in item:
                    item["relevance"] = 1 - item["_additional"]["distance"]
            
            return results
        except Exception as e:
            print(f"Audio search error: {e}")
            return []

    def search_video_frames(self, query, limit=3, user_id=None):
        """Find similar video frames"""
        try:
            # Generate query embedding
            from .Services.text_processing import TextProcessor
            query_embedding = TextProcessor().generate_embedding(query)
            
            query_builder = (
                self.client.query
                .get("VideoFrame", ["description", "frame_path", "video_path", "frame_number", "user_id"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .with_additional(["distance"])
            )
            
            # Add user filter if provided
            if user_id:
                query_builder = query_builder.with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
            
            result = query_builder.do()
            
            results = result["data"]["Get"]["VideoFrame"]
            for item in results:
                if "_additional" in item:
                    item["relevance"] = 1 - item["_additional"]["distance"]
            
            return results
        except Exception as e:
            print(f"Video search error: {e}")
            return []

    def get_user_content_count(self, user_id):
        """Get count of content items for a user"""
        try:
            counts = {}
            
            # Count text content
            text_result = (
                self.client.query.aggregate("TextContent")
                .with_meta_count()
                .with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
                .do()
            )
            counts["text"] = text_result["data"]["Aggregate"]["TextContent"][0]["meta"]["count"]
            
            # Count images
            image_result = (
                self.client.query.aggregate("ImageDescription")
                .with_meta_count()
                .with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
                .do()
            )
            counts["images"] = image_result["data"]["Aggregate"]["ImageDescription"][0]["meta"]["count"]
            
            # Count audio
            audio_result = (
                self.client.query.aggregate("AudioTranscription")
                .with_meta_count()
                .with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
                .do()
            )
            counts["audio"] = audio_result["data"]["Aggregate"]["AudioTranscription"][0]["meta"]["count"]
            
            # Count video frames
            video_result = (
                self.client.query.aggregate("VideoFrame")
                .with_meta_count()
                .with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueInt": user_id
                })
                .do()
            )
            counts["video_frames"] = video_result["data"]["Aggregate"]["VideoFrame"][0]["meta"]["count"]
            
            return counts
        except Exception as e:
            print(f"Error getting user content count: {e}")
            return {"text": 0, "images": 0, "audio": 0, "video_frames": 0}

    def delete_user_content(self, user_id):
        """Delete all content for a specific user"""
        try:
            classes = ["TextContent", "ImageDescription", "AudioTranscription", "VideoFrame"]
            deleted_counts = {}
            
            for class_name in classes:
                result = (
                    self.client.batch.delete_objects(
                        class_name=class_name,
                        where={
                            "path": ["user_id"],
                            "operator": "Equal",
                            "valueInt": user_id
                        }
                    )
                )
                deleted_counts[class_name] = result.get("successful", 0)
            
            return deleted_counts
        except Exception as e:
            print(f"Error deleting user content: {e}")
            return {}

    def health_check(self):
        """Check if the vector database is healthy"""
        try:
            result = self.client.cluster.get_nodes_status()
            return {"status": "healthy", "nodes": result}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    # Test the connection
    try:
        db = VectorDB()
        health = db.health_check()
        print(f"Connection status: {health['status']}")
        print("Successfully connected to Weaviate!")
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")