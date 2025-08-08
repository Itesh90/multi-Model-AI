import numpy as np
from transformers import pipeline
from typing import List, Dict, Any


class TextProcessor:
    def __init__(self):
        print("Loading text processing models...")
        # Text embedding model (smaller for student projects)
        self.embedding_model = pipeline(
            'feature-extraction', 
            model='sentence-transformers/all-MiniLM-L6-v2',
            device=-1  # Use CPU for students (no GPU needed)
        )
        print("✅ Embedding model loaded")
        
        # Sentiment analysis model
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=-1
        )
        print("✅ Sentiment model loaded")
        
        # Summarization model (smaller model for students)
        self.summarizer = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-12-6',
            device=-1
        )
        print("✅ Summarization model loaded")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        # Get embedding from model
        embedding = self.embedding_model(text)[0][0]
        # Convert to regular Python list
        return [float(x) for x in embedding]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        return self.sentiment_analyzer(text)[0]
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text to specified length"""
        # For very short texts, return as is
        if len(text.split()) < 20:
            return text
            
        result = self.summarizer(
            text, 
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        return result[0]['summary_text']
    
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split long text into chunks for processing"""
        # Simple chunking by sentences
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        token_count = 0
        
        for sentence in sentences:
            # Rough token estimate (1 token ~ 4 chars)
            sentence_tokens = max(1, len(sentence) // 4)
            
            if token_count + sentence_tokens > max_tokens and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                token_count = sentence_tokens
            else:
                current_chunk.append(sentence)
                token_count += sentence_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks

# Test the processor
if __name__ == "__main__":
    processor = TextProcessor()
    
    sample_text = "Multi-modal AI combines different types of data like text, images, and audio to create more powerful artificial intelligence systems. This approach mimics how humans perceive the world through multiple senses."
    
    print("\nTesting embedding generation...")
    embedding = processor.generate_embedding(sample_text)
    print(f"Embedding dimension: {len(embedding)}")
    
    print("\nTesting sentiment analysis...")
    sentiment = processor.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
    
    print("\nTesting summarization...")
    long_text = sample_text * 5  # Make it longer
    summary = processor.summarize_text(long_text)
    print(f"Original length: {len(long_text)} chars")
    print(f"Summary length: {len(summary)} chars")
    print(f"Summary: {summary}")