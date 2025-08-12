from typing import List, Dict, Any, Callable
import logging

# Attempt to import transformers pipeline; provide graceful fallback
_HAS_TRANSFORMERS = True
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - environment dependent
    _HAS_TRANSFORMERS = False
    pipeline = None

# Try to import cache_result decorator from local cache module; provide noop if missing
try:
    from .cache import cache_result  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    def cache_result(namespace: str, ttl: int = 3600) -> Callable:
        """Fallback noop cache decorator used when real cache isn't available."""
        def _decorator(fn: Callable) -> Callable:
            return fn
        return _decorator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextProcessor:
    """Handles text embeddings, sentiment analysis and summarization.

    Notes:
        - Uses HuggingFace `pipeline` when available. For student/dev setups without
          heavy dependencies the class will raise informative errors if you attempt to
          call methods that require models.
        - Embeddings/sentiment/summarization results are cached via `cache_result` when
          available. The decorator is a noop fallback if cache isn't configured.
    """

    def __init__(self):
        logger.info("Initializing TextProcessor models...")

        if not _HAS_TRANSFORMERS:
            logger.warning("transformers.pipeline not available; TextProcessor will operate in limited mode.")
            self.embedding_model = None
            self.sentiment_analyzer = None
            self.summarizer = None
            return

        # Initialize pipelines (CPU by default for student environments)
        try:
            logger.info("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)")
            self.embedding_model = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device=-1,
            )
            logger.info("✅ Embedding model loaded")

            logger.info("Loading sentiment analysis model (distilbert...)")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,
            )
            logger.info("✅ Sentiment model loaded")

            logger.info("Loading summarization model (sshleifer/distilbart-cnn-12-6)")
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1,
            )
            logger.info("✅ Summarization model loaded")

        except Exception as e:
            logger.exception("Failed to load one or more models: %s", e)
            # Set any failed component to None so calls can error gracefully
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = None
            if not hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer = None
            if not hasattr(self, 'summarizer'):
                self.summarizer = None

    @cache_result("text-embeddings", ttl=3600)
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for `text` using the configured embedding model.

        Returns:
            List[float]: embedding vector

        Raises:
            RuntimeError: if the embedding model is not loaded
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Ensure transformers are installed and models are available.")

        # The feature-extraction pipeline returns a nested list: [batch][tokens][hidden]
        raw = self.embedding_model(text)
        # Defensive handling depending on pipeline output shape
        try:
            # raw -> e.g., [[ [float...], [float...] ... ]] -> take first token vector
            emb = raw[0][0]
        except Exception:
            # Fallback: flatten and average if shape differs
            try:
                flattened = [v for seq in raw[0] for v in seq]
                emb = flattened
            except Exception as e:
                logger.exception("Unexpected embedding output format: %s", e)
                raise

        return [float(x) for x in emb]

    @cache_result("text-sentiment", ttl=3600)
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Return sentiment analysis result for the provided text.

        Raises:
            RuntimeError: if sentiment model is not loaded
        """
        if self.sentiment_analyzer is None:
            raise RuntimeError("Sentiment analyzer not loaded. Ensure transformers are installed and models are available.")

        result = self.sentiment_analyzer(text)
        # pipeline returns a list of results for batch input; return the first
        return result[0]

    @cache_result("text-summarization", ttl=3600)
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize `text` using the configured summarization pipeline.

        For short texts, the original text is returned unchanged.
        """
        if self.summarizer is None:
            raise RuntimeError("Summarizer not loaded. Ensure transformers are installed and models are available.")

        # Heuristic: for very short texts, skip summarization
        if len(text.split()) < 20:
            return text

        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False,
        )
        return result[0]["summary_text"]

    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split long text into chunks approximating `max_tokens` tokens per chunk.

        This is a simple sentence-based chunker and uses a rough token estimate.
        """
        # Basic sentence split (keeps periods)
        sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if s.strip()]
        chunks: List[str] = []
        current_chunk: List[str] = []
        token_count = 0

        for sentence in sentences:
            # Rough token estimate: 1 token ~ 4 characters
            sentence_tokens = max(1, len(sentence) // 4)

            if token_count + sentence_tokens > max_tokens and current_chunk:
                chunks.append('. '.join(current_chunk).strip() + '.')
                current_chunk = [sentence]
                token_count = sentence_tokens
            else:
                current_chunk.append(sentence)
                token_count += sentence_tokens

        if current_chunk:
            chunks.append('. '.join(current_chunk).strip() + '.')

        return chunks


# --------------------- Simple CLI / test harness ---------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small self-test")
    args = parser.parse_args()

    if args.test:
        proc = TextProcessor()

        sample_text = (
            "Multi-modal AI combines different types of data like text, images, and audio to create more "
            "powerful artificial intelligence systems. This approach mimics how humans perceive the world "
            "through multiple senses."
        )

        try:
            logger.info("Testing embedding generation...")
            emb = proc.generate_embedding(sample_text)
            logger.info("Embedding dimension: %d", len(emb))
        except Exception as e:
            logger.warning("Embedding generation skipped/failed: %s", e)

        try:
            logger.info("Testing sentiment analysis...")
            sentiment = proc.analyze_sentiment(sample_text)
            logger.info("Sentiment: %s (score: %.2f)", sentiment.get("label"), sentiment.get("score", 0.0))
        except Exception as e:
            logger.warning("Sentiment analysis skipped/failed: %s", e)

        try:
            logger.info("Testing summarization...")
            long_text = sample_text * 5
            summary = proc.summarize_text(long_text)
            logger.info("Summary length: %d chars", len(summary))
            logger.info("Summary: %s", summary)
        except Exception as e:
            logger.warning("Summarization skipped/failed: %s", e)
