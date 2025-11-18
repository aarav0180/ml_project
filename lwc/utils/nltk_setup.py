"""NLTK data download utilities."""

import nltk
import logging

logger = logging.getLogger(__name__)


def setup_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
        logger.debug("NLTK punkt_tab tokenizer found")
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
            logger.info("Downloaded NLTK punkt_tab tokenizer")
        except Exception:
            try:
                nltk.data.find('tokenizers/punkt')
                logger.debug("NLTK punkt tokenizer found")
            except LookupError:
                nltk.download('punkt', quiet=True)
                logger.info("Downloaded NLTK punkt tokenizer")

