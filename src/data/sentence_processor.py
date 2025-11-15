"""
Sentence splitting and preprocessing for advanced model.
"""
import re
import nltk
from typing import List, Tuple
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.
    
    Args:
        text: Input text string
    
    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    # Use NLTK sentence tokenizer
    sentences = nltk.sent_tokenize(text)
    
    # Clean sentences
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            cleaned.append(sent)
    
    return cleaned


def filter_sentences(sentences: List[str], min_tokens: int = 5) -> List[str]:
    """
    Remove sentences with fewer than min_tokens tokens.
    
    Args:
        sentences: List of sentence strings
        min_tokens: Minimum number of tokens required
    
    Returns:
        Filtered list of sentences
    """
    filtered = []
    for sent in sentences:
        # Simple token count (split by whitespace)
        tokens = sent.split()
        if len(tokens) >= min_tokens:
            filtered.append(sent)
    
    return filtered


def process_article(
    text: str,
    max_sentences: int = 20,
    min_tokens_per_sentence: int = 5
) -> Tuple[List[str], int]:
    """
    Process article: split into sentences, filter, and truncate/pad.
    
    Args:
        text: Full article text
        max_sentences: Maximum number of sentences to keep
        min_tokens_per_sentence: Minimum tokens per sentence
    
    Returns:
        Tuple of (list of sentences, actual number of sentences)
    """
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Filter short sentences
    sentences = filter_sentences(sentences, min_tokens=min_tokens_per_sentence)
    
    # Truncate if too many sentences
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    num_sentences = len(sentences)
    
    # Pad with empty strings if needed (will be handled in tokenization)
    if num_sentences < max_sentences:
        sentences.extend([""] * (max_sentences - num_sentences))
    
    return sentences, num_sentences


def process_articles_batch(
    texts: List[str],
    max_sentences: int = 20,
    min_tokens_per_sentence: int = 5
) -> Tuple[List[List[str]], List[int]]:
    """
    Process a batch of articles.
    
    Args:
        texts: List of article texts
        max_sentences: Maximum sentences per article
        min_tokens_per_sentence: Minimum tokens per sentence
    
    Returns:
        Tuple of (list of sentence lists, list of actual sentence counts)
    """
    processed_sentences = []
    sentence_counts = []
    
    for text in texts:
        sentences, count = process_article(
            text,
            max_sentences=max_sentences,
            min_tokens_per_sentence=min_tokens_per_sentence
        )
        processed_sentences.append(sentences)
        sentence_counts.append(count)
    
    return processed_sentences, sentence_counts

