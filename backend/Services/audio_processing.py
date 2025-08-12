import io
import os
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, ContextManager
from contextlib import contextmanager

import numpy as np
import torch
import librosa
import soundfile as sf
import whisper

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AudioConfig:
    """Configuration class for audio processing settings"""
    
    # Model settings
    whisper_model_size: str = "tiny"
    device: str = "cpu"
    
    # Audio processing settings
    target_sample_rate: int = 16000
    embedding_dimension: int = 512
    mfcc_coefficients: int = 13
    
    # File handling
    temp_file_suffix: str = ".wav"
    max_audio_duration: float = 300.0  # 5 minutes max
    supported_formats: tuple = ('.wav', '.mp3', '.m4a', '.flac')
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.whisper_model_size not in ["tiny", "base", "small", "medium", "large"]:
            raise ValueError(f"Invalid Whisper model size: {self.whisper_model_size}")
        
        if self.target_sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    pass

class AudioFormatError(AudioProcessingError):
    """Raised when audio format is invalid or unsupported"""
    pass

class TranscriptionError(AudioProcessingError):
    """Raised when audio transcription fails"""
    pass

class EmbeddingError(AudioProcessingError):
    """Raised when audio embedding generation fails"""
    pass

class ModelLoadError(AudioProcessingError):
    """Raised when model loading fails"""
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logger(
    name: str,
    config: Optional[AudioConfig] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration
    
    Args:
        name: Logger name
        config: Audio configuration object
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = AudioConfig()
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(config.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

@contextmanager
def temp_file_manager(suffix: str = ".wav"):
    """
    Context manager for temporary file handling
    
    Args:
        suffix: File suffix for the temporary file
        
    Yields:
        Path to temporary file
    """
    temp_file = None
    temp_path = None
    
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def validate_audio_format(file_path: Union[str, Path], config: AudioConfig) -> bool:
    """
    Validate if the audio file format is supported
    
    Args:
        file_path: Path to the audio file
        config: Audio configuration
        
    Returns:
        True if format is supported
        
    Raises:
        AudioFormatError: If format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise AudioFormatError(f"Audio file does not exist: {file_path}")
    
    if file_path.suffix.lower() not in config.supported_formats:
        raise AudioFormatError(
            f"Unsupported audio format: {file_path.suffix}. "
            f"Supported formats: {config.supported_formats}"
        )
    
    return True

# =============================================================================
# CORE PROCESSING CLASSES
# =============================================================================

class WhisperModel:
    """Wrapper for Whisper speech-to-text model"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = setup_logger(f"{__name__}.WhisperModel")
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model ({self.config.whisper_model_size})...")
            self.model = whisper.load_model(self.config.whisper_model_size)
            self.model.eval()
            
            # Move to appropriate device
            device = torch.device(self.config.device)
            self.model = self.model.to(device)
            
            self.logger.info("✅ Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription results with text, language, and segments
            
        Raises:
            TranscriptionError: If transcription fails
        """
        if not self.model:
            raise TranscriptionError("Whisper model not loaded")
        
        try:
            result = self.model.transcribe(audio_path)
            
            self.logger.info(f"Transcribed audio: {result['text'][:50]}...")
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

class AudioPreprocessor:
    """Handles audio preprocessing tasks"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = setup_logger(f"{__name__}.AudioPreprocessor")
    
    def preprocess_audio(
        self, 
        audio_path: str, 
        target_sample_rate: int = None
    ) -> np.ndarray:
        """
        Preprocess audio data for model input
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate for processing
            
        Returns:
            Audio waveform as numpy array
            
        Raises:
            AudioFormatError: If audio preprocessing fails
        """
        target_sample_rate = target_sample_rate or self.config.target_sample_rate
        
        try:
            # Load audio with librosa
            audio, sample_rate = librosa.load(
                audio_path, 
                sr=target_sample_rate
            )
            
            # Validate audio duration
            duration = len(audio) / sample_rate
            if duration > self.config.max_audio_duration:
                self.logger.warning(
                    f"Audio duration ({duration:.2f}s) exceeds maximum "
                    f"({self.config.max_audio_duration}s). Truncating..."
                )
                audio = audio[:int(self.config.max_audio_duration * sample_rate)]
            
            # Normalize audio
            audio = self.normalize_audio(audio)
            
            self.logger.debug(
                f"Preprocessed audio: {len(audio)} samples, {sample_rate} Hz, "
                f"{duration:.2f}s duration"
            )
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {str(e)}")
            raise AudioFormatError(f"Invalid audio format: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

class FeatureExtractor:
    """Extracts features from audio for embedding generation"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = setup_logger(f"{__name__}.FeatureExtractor")
    
    def extract_mfcc_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate of audio
            
        Returns:
            MFCC feature matrix
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.config.mfcc_coefficients
            )
            return mfccs
            
        except Exception as e:
            self.logger.error(f"MFCC extraction failed: {str(e)}")
            raise EmbeddingError(f"Failed to extract MFCC features: {str(e)}")
    
    def generate_embedding(self, audio: np.ndarray, sample_rate: int) -> List[float]:
        """
        Generate audio embedding using MFCC features
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate of audio
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Extract MFCC features
            mfccs = self.extract_mfcc_features(audio, sample_rate)
            
            # Compute statistics across time axis for richer representation
            mean_features = np.mean(mfccs, axis=1)
            std_features = np.std(mfccs, axis=1)
            
            # Combine mean and std for a more comprehensive embedding
            embedding = np.concatenate([mean_features, std_features]).tolist()
            
            # Ensure consistent embedding size
            embedding = self._pad_or_truncate_embedding(embedding)
            
            self.logger.info(f"Generated audio embedding of dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Audio embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate audio embedding: {str(e)}")
    
    def _pad_or_truncate_embedding(self, embedding: List[float]) -> List[float]:
        """
        Ensure embedding has consistent dimension
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Padded or truncated embedding
        """
        target_dim = self.config.embedding_dimension
        
        if len(embedding) < target_dim:
            # Pad with zeros
            embedding.extend([0.0] * (target_dim - len(embedding)))
        elif len(embedding) > target_dim:
            # Truncate
            embedding = embedding[:target_dim]
        
        return embedding

# =============================================================================
# MAIN AUDIO PROCESSOR CLASS
# =============================================================================

class AudioProcessor:
    """
    Main class for handling audio processing tasks
    
    This class provides a unified interface for audio transcription and 
    embedding generation, combining Whisper for speech-to-text and 
    MFCC-based features for audio embeddings.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio processor with configuration
        
        Args:
            config: Audio configuration object. If None, uses default config.
        """
        self.config = config or AudioConfig()
        self.logger = setup_logger(f"{__name__}.AudioProcessor")
        
        self.logger.info("Initializing audio processing components...")
        
        try:
            # Initialize components
            self.whisper_model = WhisperModel(self.config)
            self.preprocessor = AudioPreprocessor(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            
            self.logger.info("✅ Audio processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio processor: {str(e)}")
            raise AudioProcessingError(f"Initialization failed: {str(e)}")
    
    def preprocess_audio(self, audio_data: bytes, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data for model input (legacy method for compatibility)
        
        Args:
            audio_data: Raw audio bytes from upload
            target_sample_rate: Target sample rate for processing
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            with temp_file_manager(self.config.temp_file_suffix) as temp_path:
                # Write audio data to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                # Validate and preprocess
                validate_audio_format(temp_path, self.config)
                audio = self.preprocessor.preprocess_audio(temp_path, target_sample_rate)
                
                self.logger.debug(f"Preprocessed audio: {len(audio)} samples, {target_sample_rate} Hz")
                return audio
                
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {str(e)}")
            raise AudioFormatError(f"Invalid audio format: {str(e)}")
    
    def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe speech to text
        
        Args:
            audio_data: Raw audio bytes from upload
            
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected language
                - segments: List of transcription segments with timestamps
            
        Raises:
            AudioProcessingError: If transcription fails
        """
        try:
            with temp_file_manager(self.config.temp_file_suffix) as temp_path:
                # Write audio data to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                # Validate audio format
                validate_audio_format(temp_path, self.config)
                
                # Transcribe using Whisper
                result = self.whisper_model.transcribe(temp_path)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Audio transcription failed: {str(e)}")
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    def generate_audio_embedding(self, audio_data: bytes) -> List[float]:
        """
        Generate embedding for audio
        
        Args:
            audio_data: Raw audio bytes from upload
            
        Returns:
            Embedding vector as a list of floats with dimension specified in config
            
        Raises:
            AudioProcessingError: If embedding generation fails
        """
        try:
            with temp_file_manager(self.config.temp_file_suffix) as temp_path:
                # Write audio data to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                # Validate audio format
                validate_audio_format(temp_path, self.config)
                
                # Preprocess audio
                audio = self.preprocessor.preprocess_audio(
                    temp_path, 
                    self.config.target_sample_rate
                )
                
                # Generate embedding
                embedding = self.feature_extractor.generate_embedding(
                    audio, 
                    self.config.target_sample_rate
                )
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Audio embedding generation failed: {str(e)}")
            raise AudioProcessingError(f"Embedding generation failed: {str(e)}")
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio for both transcription and embedding generation
        
        Args:
            audio_data: Raw audio bytes from upload
            
        Returns:
            Dictionary containing:
                - transcription: Full transcription results
                - embedding: Audio embedding vector
                - embedding_dimension: Length of embedding vector
                - metadata: Additional processing metadata
        """
        try:
            self.logger.info("Starting comprehensive audio processing...")
            
            # Process transcription and embedding
            transcription = self.transcribe_audio(audio_data)
            embedding = self.generate_audio_embedding(audio_data)
            
            # Compile results
            result = {
                "transcription": transcription,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "metadata": {
                    "config": {
                        "whisper_model": self.config.whisper_model_size,
                        "sample_rate": self.config.target_sample_rate,
                        "embedding_dimension": self.config.embedding_dimension
                    },
                    "processing_info": {
                        "text_length": len(transcription["text"]),
                        "detected_language": transcription["language"],
                        "num_segments": len(transcription.get("segments", []))
                    }
                }
            }
            
            self.logger.info("✅ Audio processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive audio processing failed: {str(e)}")
            raise AudioProcessingError(f"Processing failed: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats
        
        Returns:
            List of supported file extensions
        """
        return list(self.config.supported_formats)
    
    def get_config(self) -> AudioConfig:
        """
        Get current configuration
        
        Returns:
            Current AudioConfig object
        """
        return self.config

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_audio_processor(
    model_size: str = "tiny",
    embedding_dim: int = 512,
    sample_rate: int = 16000,
    device: str = "cpu"
) -> AudioProcessor:
    """
    Convenience function to create an AudioProcessor with common settings
    
    Args:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        embedding_dim: Dimension of audio embeddings
        sample_rate: Target sample rate for processing
        device: Computing device ("cpu" or "cuda")
        
    Returns:
        Configured AudioProcessor instance
    """
    config = AudioConfig(
        whisper_model_size=model_size,
        embedding_dimension=embedding_dim,
        target_sample_rate=sample_rate,
        device=device
    )
    return AudioProcessor(config)

# =============================================================================
# TESTING AND EXAMPLE USAGE
# =============================================================================

def run_audio_processor_test():
    """Run basic tests for the audio processor"""
    print("🎵 Audio Processor Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration
        print("1. Testing configuration...")
        config = AudioConfig(
            whisper_model_size="tiny",
            embedding_dimension=128,  # Smaller for testing
            target_sample_rate=16000
        )
        print("✅ Configuration created successfully")
        
        # Test processor initialization
        print("\n2. Testing processor initialization...")
        processor = AudioProcessor(config)
        print("✅ AudioProcessor initialized successfully")
        
        # Test supported formats
        print(f"\n3. Supported formats: {processor.get_supported_formats()}")
        
        # Test with sample file if available
        test_files = ["test.wav", "test_audio.wav", "sample.wav"]
        test_file = None
        
        for filename in test_files:
            if Path(filename).exists():
                test_file = filename
                break
        
        if test_file:
            print(f"\n4. Testing with {test_file}")
            
            with open(test_file, "rb") as f:
                audio_data = f.read()
            
            # Test transcription
            print("   📝 Testing transcription...")
            transcription = processor.transcribe_audio(audio_data)
            print(f"   Text: {transcription['text'][:100]}...")
            print(f"   Language: {transcription['language']}")
            
            # Test embedding generation
            print("   🔢 Testing embedding generation...")
            embedding = processor.generate_audio_embedding(audio_data)
            print(f"   Embedding dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            
            # Test combined processing
            print("   🔄 Testing combined processing...")
            result = processor.process_audio(audio_data)
            print(f"   Result keys: {list(result.keys())}")
            print(f"   Metadata: {result['metadata']['processing_info']}")
            
        else:
            print("\n4. ⚠️  No test audio file found")
            print("   Create a 'test.wav' file to run audio processing tests")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run tests
    run_audio_processor_test()
    
    # Example usage
    print("\n" + "="*50)
    print("📖 Example Usage:")
    print("="*50)
    
    example_code = '''
# Basic usage
from audio_processor import AudioProcessor, AudioConfig

# Create processor with default settings
processor = AudioProcessor()

# Or with custom configuration
config = AudioConfig(
    whisper_model_size="base",
    embedding_dimension=256,
    target_sample_rate=22050
)
processor = AudioProcessor(config)

# Process audio file
with open("your_audio.wav", "rb") as f:
    audio_data = f.read()

# Get transcription only
transcription = processor.transcribe_audio(audio_data)
print(f"Transcribed text: {transcription['text']}")

# Get embedding only
embedding = processor.generate_audio_embedding(audio_data)
print(f"Embedding shape: {len(embedding)}")

# Get both transcription and embedding
result = processor.process_audio(audio_data)
print(f"Complete result: {result}")
'''
    
    print(example_code)