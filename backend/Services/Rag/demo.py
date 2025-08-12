import logging

from .config import RAGConfig
from .rag_services import RAGService
from .models import MultiModalRequest


def setup_logging() -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Main function for testing the RAG service"""
    setup_logging()

    try:
        # Initialize RAG service
        config = RAGConfig()
        rag = RAGService(config)

        # Test basic response generation
        print("\nTesting basic response generation...")
        response = rag.generate_response(
            "What is the main topic of the documents I've uploaded?"
        )
        print(f"Response: {response.result}")
        print(f"Sources: {len(response.sources)} documents")
        if response.processing_time is not None:
            print(f"Processing time: {response.processing_time:.2f}s")

        # Test multi-modal response
        print("\nTesting multi-modal response...")
        multimodal_request = MultiModalRequest(
            query="Describe what's happening in this scene",
            image_data=b"dummy_image_data",
            user_id=123,
        )

        multi_response = rag.generate_multimodal_response(multimodal_request)
        print(f"Multi-modal response: {multi_response.result}")

    except Exception as e:
        logging.error(f"Application error: {e}")


if __name__ == "__main__":
    main()


