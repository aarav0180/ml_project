"""
Data loading and preprocessing for advanced model with sentence-level processing.
"""
import torch
from typing import List
from transformers import BertTokenizerFast

from .data_loader import load_data, prepare_data
from .sentence_processor import process_articles_batch


class AdvancedDataProcessor:
    """
    Processes data for advanced model with sentence-level tokenization.
    """
    
    def __init__(
        self,
        tokenizer_name='bert-base-uncased',
        max_sentences=20,
        max_tokens_per_sentence=64,
        max_tokens_per_article=256,
        min_tokens_per_sentence=5
    ):
        """
        Initialize data processor.
        
        Args:
            tokenizer_name: BERT tokenizer name
            max_sentences: Maximum sentences per article
            max_tokens_per_sentence: Maximum tokens per sentence
            max_tokens_per_article: Maximum tokens for full article
            min_tokens_per_sentence: Minimum tokens per sentence
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_sentences = max_sentences
        self.max_tokens_per_sentence = max_tokens_per_sentence
        self.max_tokens_per_article = max_tokens_per_article
        self.min_tokens_per_sentence = min_tokens_per_sentence
    
    def process_texts(
        self,
        texts: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], List[int]]:
        """
        Process texts into sentence-level and article-level tokens.
        
        Args:
            texts: List of article texts
        
        Returns:
            Tuple of:
                - sentence_input_ids: (batch_size, max_sentences, max_tokens_per_sentence)
                - sentence_attention_mask: (batch_size, max_sentences, max_tokens_per_sentence)
                - article_input_ids: (batch_size, max_tokens_per_article)
                - article_attention_mask: (batch_size, max_tokens_per_article)
                - sentence_texts: List of lists of sentence strings (batch_size, num_sentences)
                - sentence_counts: List of actual sentence counts per article
        """
        # Process articles into sentences
        sentence_lists, sentence_counts = process_articles_batch(
            texts,
            max_sentences=self.max_sentences,
            min_tokens_per_sentence=self.min_tokens_per_sentence
        )
        
        batch_size = len(texts)
        
        # Tokenize sentences
        sentence_input_ids = []
        sentence_attention_mask = []
        
        for sentences in sentence_lists:
            # Tokenize each sentence
            sentence_tokens = self.tokenizer.batch_encode_plus(
                sentences,
                max_length=self.max_tokens_per_sentence,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            sentence_input_ids.append(sentence_tokens['input_ids'])
            sentence_attention_mask.append(sentence_tokens['attention_mask'])
        
        # Stack into tensors
        sentence_input_ids = torch.stack(sentence_input_ids)  # (batch_size, max_sentences, max_tokens_per_sentence)
        sentence_attention_mask = torch.stack(sentence_attention_mask)  # (batch_size, max_sentences, max_tokens_per_sentence)
        
        # Tokenize full articles
        article_tokens = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_tokens_per_article,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        article_input_ids = article_tokens['input_ids']  # (batch_size, max_tokens_per_article)
        article_attention_mask = article_tokens['attention_mask']  # (batch_size, max_tokens_per_article)
        
        return (
            sentence_input_ids,
            sentence_attention_mask,
            article_input_ids,
            article_attention_mask,
            sentence_lists,  # Return raw sentence texts
            sentence_counts
        )
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: List[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        Prepare complete dataset.
        
        Args:
            texts: List of article texts
            labels: List of labels
        
        Returns:
            Tuple of all tensors and sentence texts needed for training
        """
        sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, _ = self.process_texts(texts)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels_tensor

