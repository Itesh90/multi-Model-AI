import io
import numpy as np
import torch
import librosa
import soundfile as sf
import whisper
from typing import Dict, Any, List, Optional
import logging
import os
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing tasks for the multi-modal platform"""
    
    def __init__(self):
        """Initialize audio processing models"""
        logger.info("Initializing audio processing models...")
        
        try:
            # Initialize Whisper for speech-to-text
            logger.info("Loading Whisper model (tiny)...")
            self.whisper_model = whisper.load_model("tiny")
            logger.info("✅ Whisper model loaded successfully")
            
            # Set to evaluation mode
            self.whisper_model.eval()
            
            # Device setup
            self.device = torch.device("cpu")  # CPU for students
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: bytes, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data for model input
        
        Args:
            audio_data: Raw audio bytes from upload
            target_sample_rate: Target sample rate for processing
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            
            # Load audio with librosa
            audio, sample_rate = librosa.load(temp_path, sr=target_sample_rate)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.debug(f"Preprocessed audio: {len(audio)} samples, {sample_rate} Hz")
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise ValueError(f"Invalid audio format: {str(e)}")
    
    def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe speech to text
        
        Args:
            audio_data: Raw audio bytes from upload
            
        Returns:
            Transcription results
        """
        try:
            # Create temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            logger.info(f"Transcribed audio: {result['text'][:50]}...")
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def generate_audio_embedding(self, audio_data: bytes) -> List[float]:
        """
        Generate embedding for audio (simplified for students)
        
        Args:
            audio_data: Raw audio bytes from upload
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            # For students without advanced models, we'll use a simpler approach
            # In a real implementation, you'd use a proper audio embedding model
            
            # Preprocess audio
            audio = self.preprocess_audio(audio_data)
            
            # Simple feature extraction (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
            embedding = np.mean(mfccs, axis=1).tolist()
            
            # Pad to a consistent size
            while len(embedding) < 512:
                embedding.append(0.0)
                
            logger.info(f"Generated audio embedding of dimension: {len(embedding)}")
            return embedding[:512]  # Truncate to 512 dimensions
            
        except Exception as e:
            logger.error(f"Audio embedding generation failed: {str(e)}")
            raise

# Test the processor
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    processor = AudioProcessor()
    
    # Try to load a test audio file
    try:
        with open("test.wav", "rb") as f:
            audio_data = f.read()
        
        print("\nTesting audio transcription...")
        transcription = processor.transcribe_audio(audio_data)
        print(f"Transcribed text: {transcription['text']}")
        print(f"Detected language: {transcription['language']}")
        
        print("\nTesting audio embedding generation...")
        embedding = processor.generate_audio_embedding(audio_data)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
    except FileNotFoundError:
        print("❌ Test failed: Please create a 'test.wav' file in this directory")
        print("You can use any WAV audio file for testing")