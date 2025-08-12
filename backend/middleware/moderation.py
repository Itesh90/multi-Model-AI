import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ContentModerator:
    """Basic content moderation for student projects.

    Notes:
        - This is a simple keyword-based moderator intended for learning/demo purposes.
        - For production use, replace with a model-based or third-party moderation service.
    """

    def __init__(self) -> None:
        """Initialize moderation system with simple rules."""
        logger.info("Initializing content moderation system")

        # Simple keyword-based moderation (for students)
        self.prohibited_words: List[str] = [
            "hate", "racist", "offensive", "abuse", "violence",
            "nsfw", "porn", "explicit", "sexual", "nude",
        ]

        # Categories we can detect
        self.categories: List[str] = ["hate", "violence", "sexual", "self-harm"]

        logger.info(f"Initialized with {len(self.prohibited_words)} prohibited words")

    def moderate_text(self, text: str) -> Dict[str, Any]:
        """Moderate text content.

        Args:
            text: Text to moderate

        Returns:
            Dictionary with moderation results
        """
        original_text = text
        text_lower = text.lower()
        detected_categories: List[str] = []
        matched_words: List[str] = []

        # Check for prohibited words
        for word in self.prohibited_words:
            if word in text_lower:
                category = self._get_category_for_word(word)
                if category not in detected_categories:
                    detected_categories.append(category)
                matched_words.append(word)

        # Calculate overall severity
        severity = "low"
        if len(detected_categories) >= 2:
            severity = "medium"
        if len(detected_categories) >= 3:
            severity = "high"

        result = {
            "is_safe": len(detected_categories) == 0,
            "severity": severity,
            "categories": detected_categories,
            "matched_words": matched_words,
            "original_length": len(original_text),
            "censored_text": self._censor_text(original_text) if detected_categories else original_text,
        }

        if not result["is_safe"]:
            logger.warning("Content moderation flagged text: %s", result)

        return result

    def _get_category_for_word(self, word: str) -> str:
        """Map prohibited word to category."""
        if word in ["hate", "racist"]:
            return "hate"
        if word in ["violence", "abuse"]:
            return "violence"
        if word in ["nsfw", "porn", "explicit", "sexual", "nude"]:
            return "sexual"
        return "other"

    def _censor_text(self, text: str) -> str:
        """Censor detected prohibited words by replacing inner characters with asterisks.

        Keeps the first character of each censored word for readability.
        """
        censored = text
        for word in self.prohibited_words:
            # Use case-insensitive whole-word replacement
            pattern = r"\b" + re.escape(word) + r"\b"
            replacement = word[0] + "*" * (len(word) - 1)
            censored = re.sub(pattern, replacement, censored, flags=re.IGNORECASE)
        return censored

    def moderate_image(self, image_description: str) -> Dict[str, Any]:
        """Moderate image based on its textual description by reusing text moderation."""
        return self.moderate_text(image_description)


# --------------------- Simple CLI / test harness ---------------------
if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    moderator = ContentModerator()

    # Test safe text
    print("\nTesting safe text...")
    safe_text = "This is a perfectly normal and safe message."
    result_safe = moderator.moderate_text(safe_text)
    print(f"Safe text result: {result_safe}")

    # Test unsafe text
    print("\nTesting unsafe text...")
    unsafe_text = "This message contains hate speech and explicit content that is not appropriate."
    result_unsafe = moderator.moderate_text(unsafe_text)
    print(f"Unsafe text result: {result_unsafe}")
    print(f"Censored text: {result_unsafe['censored_text']}")