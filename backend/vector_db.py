import os
import weaviate
from weaviate.auth import AuthApiKey

class VectorDB:
    def __init__(self):
        # In a real app, these would come from environment variables
        self.url = "omgvdcpdtzc3xahbyuczeg.c0.asia-southeast1.gcp.weaviate.cloud"  # REPLACE WITH YOUR URL
        self.api_key = "WG01VkFDUitlbFZCS291NF92Q0ZFcXlQUkN3UDdqV214STlwL0thbnBuRlZiZmVpSUFxNFlLOURGYTRVPV92MjAw"  # REPLACE WITH YOUR API KEY
        
        self.client = weaviate.Client(
            url=self.url,
            auth_client_secret=AuthApiKey(api_key=self.api_key),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")  # If using OpenAI
            }
        )
        self._setup_schema()
    
    def _setup_schema(self):
        """Create schema if it doesn't exist"""
        try:
            # Check if schema already exists
            if self.client.schema.exists("ImageDescription"):
                return
                
            # Define schema
            class_obj = {
                "class": "ImageDescription",
                "description": "Descriptions of images for semantic search",
                "vectorizer": "text2vec-openai",  # Using OpenAI embeddings
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "0.0.1",
                        "type": "text"
                    }
                },
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
                    }
                ]
            }
            
            # Create the class
            self.client.schema.create_class(class_obj)
            print("Schema created successfully!")
        except Exception as e:
            print(f"Error setting up schema: {e}")


         # Add TextContent class if it doesn't exist
        if not self.client.schema.exists("TextContent"):
          text_class = {
            "class": "TextContent",
            "description": "General text content for semantic search",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "The main text content"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata about the text"
                }
            ]
        }
        self.client.schema.create_class(text_class)
     
        # Add to _setup_schema method
        if not self.client.schema.exists("AudioTranscription"):
          audio_class = {
         "class": "AudioTranscription",
         "description": "Transcriptions of audio content",
         "vectorizer": "none",  # We provide our own vectors
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
                "dataType": ["string"],
                "description": "Detected language of the audio"
            }
        ]
    }
    self.client.schema.create_class(audio_class)

    
    
    def add_image_description(self, description, image_path, user_id, embedding=None):
     """Add an image description to the vector database with embedding"""
    data_object = {
        "description": description,
        "image_path": image_path,
        "user_id": user_id
    }
    
    try:
        # If embedding is provided, use it
        if embedding:
            self.client.data_object.create(
                data_object,
                "ImageDescription",
                vector=embedding
            )
        else:
            self.client.data_object.create(
                data_object,
                "ImageDescription"
            )
        return True
    except Exception as e:
        print(f"Error adding to vector DB: {e}")
        return False
    def search_similar(self, query, limit=3):
        """Find similar image descriptions"""
        try:
            result = (
                self.client.query
                .get("ImageDescription", ["description", "image_path", "user_id"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            return result["data"]["Get"]["ImageDescription"]
        except Exception as e:
            print(f"Search error: {e}")
            return []
        

        def add_text_with_metadata(self, text, metadata=None):
    """Add text with metadata to vector database"""
    # In a real implementation, you'd create a proper schema
    # For our student project, we'll use a simple approach
    data_object = {
        "description": text,
        "metadata": metadata or {}
    }
    
    try:
        self.client.data_object.create(
            data_object,
            "TextContent"  # You'd need to create this class first
        )
        return True
    except Exception as e:
        print(f"Error adding to vector DB: {e}")
        return False


# Update vector_db.py to add this method
   def search_similar_by_vector(self, query_embedding, limit=3):
    """Find similar image descriptions by vector"""
    try:
        result = (
            self.client.query
            .get("ImageDescription", ["description", "image_path", "user_id"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
            .do()
        )
        return result["data"]["Get"]["ImageDescription"]
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


    def add_audio_transcription(self, transcription, audio_path, user_id, embedding, language):
    """Add audio transcription to vector database"""
    data_object = {
        "transcription": transcription,
        "audio_path": audio_path,
        "user_id": user_id,
        "language": language
    }
    
    try:
        self.client.data_object.create(
            data_object,
            "AudioTranscription",
            vector=embedding
        )
        return True
    except Exception as e:
        print(f"Error adding audio to vector DB: {e}")
        return False
    
    def add_audio_transcription(self, transcription, audio_path, user_id, embedding, language):
    """Add audio transcription to vector database"""
    data_object = {
        "transcription": transcription,
        "audio_path": audio_path,
        "user_id": user_id,
        "language": language
    }
    
    try:
        self.client.data_object.create(
            data_object,
            "AudioTranscription",
            vector=embedding
        )
        return True
    except Exception as e:
        print(f"Error adding audio to vector DB: {e}")
        return False

def search_similar_audio(self, query_embedding, limit=3):
    """Find similar audio transcriptions by vector"""
    try:
        result = (
            self.client.query
            .get("AudioTranscription", ["transcription", "audio_path", "user_id", "language"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
            .do()
        )
        return result["data"]["Get"]["AudioTranscription"]
    except Exception as e:
        print(f"Audio search error: {e}")
        return []

# Test the connection
if __name__ == "__main__":
    db = VectorDB()
    print("Successfully connected to Weaviate!")