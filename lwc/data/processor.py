"""Data processing utilities for sentence-level tokenization."""

from typing import List, Tuple
import torch
import pandas as pd
import nltk
from transformers import BertTokenizerFast
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []
    sentences = nltk.sent_tokenize(text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned


def filter_sentences(sentences: List[str], min_tokens: int = 5) -> List[str]:
    """
    Remove sentences with fewer than min_tokens tokens.
    
    Args:
        sentences: List of sentences
        min_tokens: Minimum number of tokens required
        
    Returns:
        Filtered list of sentences
    """
    filtered = []
    for sent in sentences:
        tokens = sent.split()
        if len(tokens) >= min_tokens:
            filtered.append(sent)
    return filtered


def process_article(
    text: str,
    max_sentences: int = 12,
    min_tokens_per_sentence: int = 5
) -> Tuple[List[str], int]:
    """
    Process article: split into sentences, filter, and truncate/pad.
    
    Args:
        text: Article text
        max_sentences: Maximum number of sentences to keep
        min_tokens_per_sentence: Minimum tokens per sentence
        
    Returns:
        Tuple of (processed_sentences, num_sentences)
    """
    sentences = split_into_sentences(text)
    sentences = filter_sentences(sentences, min_tokens=min_tokens_per_sentence)

    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    num_sentences = len(sentences)

    if num_sentences < max_sentences:
        sentences.extend([""] * (max_sentences - num_sentences))

    return sentences, num_sentences


def process_articles_batch(
    texts: List[str],
    max_sentences: int = 12,
    min_tokens_per_sentence: int = 5,
    show_progress: bool = True
) -> Tuple[List[List[str]], List[int]]:
    """
    Process a batch of articles.
    
    Args:
        texts: List of article texts
        max_sentences: Maximum number of sentences per article
        min_tokens_per_sentence: Minimum tokens per sentence
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (processed_sentences_list, sentence_counts)
    """
    processed_sentences = []
    sentence_counts = []

    iterator = tqdm(texts, desc="Processing articles", unit="article") if show_progress else texts
    for text in iterator:
        sentences, count = process_article(
            text,
            max_sentences=max_sentences,
            min_tokens_per_sentence=min_tokens_per_sentence
        )
        processed_sentences.append(sentences)
        sentence_counts.append(count)

    return processed_sentences, sentence_counts


class AdvancedDataProcessor:
    """Processes data for advanced model with sentence-level tokenization."""

    def __init__(
        self,
        tokenizer_name: str = 'bert-base-uncased',
        max_sentences: int = 12,
        max_tokens_per_sentence: int = 48,
        max_tokens_per_article: int = 256,
        min_tokens_per_sentence: int = 5
    ):
        """
        Initialize the data processor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_sentences: Maximum number of sentences per article
            max_tokens_per_sentence: Maximum tokens per sentence
            max_tokens_per_article: Maximum tokens per article
            min_tokens_per_sentence: Minimum tokens per sentence
        """
        logger.info(f"Initializing data processor with tokenizer: {tokenizer_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_sentences = max_sentences
        self.max_tokens_per_sentence = max_tokens_per_sentence
        self.max_tokens_per_article = max_tokens_per_article
        self.min_tokens_per_sentence = min_tokens_per_sentence

    def process_texts(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], List[int]]:
        """
        Process texts into sentence-level and article-level tokens.
        
        Args:
            texts: List of article texts
            
        Returns:
            Tuple of (sentence_ids, sentence_mask, article_ids, article_mask, sentence_lists, sentence_counts)
        """
        # Ensure all texts are strings and handle None/NaN
        cleaned_texts = []
        for text in texts:
            try:
                if text is None:
                    cleaned_texts.append("")
                elif isinstance(text, float) and (text != text):  # NaN check (NaN != NaN)
                    cleaned_texts.append("")
                elif pd.isna(text):
                    cleaned_texts.append("")
                else:
                    cleaned_texts.append(str(text))
            except (TypeError, ValueError):
                # Fallback for any other type issues
                cleaned_texts.append("")
        
        logger.info(f"Processing {len(cleaned_texts)} articles...")
        sentence_lists, sentence_counts = process_articles_batch(
            cleaned_texts,
            max_sentences=self.max_sentences,
            min_tokens_per_sentence=self.min_tokens_per_sentence,
            show_progress=True
        )

        batch_size = len(cleaned_texts)

        # Tokenize sentences with progress bar
        logger.info("Tokenizing sentences...")
        sentence_input_ids = []
        sentence_attention_mask = []

        for sentences in tqdm(sentence_lists, desc="Tokenizing sentences", unit="article"):
            # Ensure all sentences are strings
            cleaned_sentences = [str(s) if s is not None else "" for s in sentences]
            sentence_tokens = self.tokenizer.batch_encode_plus(
                cleaned_sentences,
                max_length=self.max_tokens_per_sentence,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            sentence_input_ids.append(sentence_tokens['input_ids'])
            sentence_attention_mask.append(sentence_tokens['attention_mask'])

        sentence_input_ids = torch.stack(sentence_input_ids)
        sentence_attention_mask = torch.stack(sentence_attention_mask)

        # Tokenize full articles
        logger.info("Tokenizing full articles...")
        article_tokens = self.tokenizer.batch_encode_plus(
            cleaned_texts,
            max_length=self.max_tokens_per_article,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        article_input_ids = article_tokens['input_ids']
        article_attention_mask = article_tokens['attention_mask']

        return (
            sentence_input_ids,
            sentence_attention_mask,
            article_input_ids,
            article_attention_mask,
            sentence_lists,
            sentence_counts
        )

    def prepare_dataset(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        Prepare complete dataset.
        
        Args:
            texts: List of article texts
            labels: List of labels
            
        Returns:
            Tuple of (sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels_tensor)
        """
        sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, _ = self.process_texts(texts)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels_tensor

