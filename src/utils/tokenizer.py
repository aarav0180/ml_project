"""
Tokenization utilities for BERT model.
"""
import torch
from transformers import BertTokenizerFast


class Tokenizer:
    """
    Wrapper class for BERT tokenizer with convenient methods.
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Name of the BERT model
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model_name = model_name
    
    def encode(self, texts, max_length=15, padding="max_length", truncation=True):
        """
        Encode texts into token IDs and attention masks.
        
        Args:
            texts: List of text strings or pandas Series
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if isinstance(texts, (list, tuple)):
            text_list = texts
        else:
            text_list = texts.tolist()
        
        return self.tokenizer.batch_encode_plus(
            text_list,
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
    
    def encode_to_tensors(self, texts, max_length=15, device='cpu'):
        """
        Encode texts and convert to PyTorch tensors.
        
        Args:
            texts: List of text strings or pandas Series
            max_length: Maximum sequence length
            device: Device to place tensors on
        
        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        tokens = self.encode(texts, max_length=max_length)
        
        input_ids = torch.tensor(tokens['input_ids']).to(device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(device)
        
        return input_ids, attention_mask

