"""Dataset class for advanced model."""

from typing import List
import torch
from torch.utils.data import Dataset


class AdvancedDataset(Dataset):
    """Dataset class for advanced model."""

    def __init__(
        self,
        sentence_ids: torch.Tensor,
        sentence_mask: torch.Tensor,
        article_ids: torch.Tensor,
        article_mask: torch.Tensor,
        sentence_texts: List[List[str]],
        labels: torch.Tensor
    ):
        """
        Initialize the dataset.
        
        Args:
            sentence_ids: Sentence token IDs
            sentence_mask: Sentence attention masks
            article_ids: Article token IDs
            article_mask: Article attention masks
            sentence_texts: List of lists of sentence strings
            labels: Labels tensor
        """
        self.sentence_ids = sentence_ids
        self.sentence_mask = sentence_mask
        self.article_ids = article_ids
        self.article_mask = article_mask
        self.sentence_texts = sentence_texts
        self.labels = labels

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        return (
            self.sentence_ids[idx],
            self.sentence_mask[idx],
            self.article_ids[idx],
            self.article_mask[idx],
            self.sentence_texts[idx],
            self.labels[idx]
        )


def custom_collate_fn(batch_items):
    """
    Custom collate function to handle the list of strings.
    
    Args:
        batch_items: List of tuples from dataset
        
    Returns:
        Batched tensors and lists
    """
    # Separate the items into their respective lists
    s_ids, s_mask, a_ids, a_mask, s_texts_list, labs = zip(*batch_items)

    # Stack the tensors
    s_ids_batch = torch.stack(s_ids)
    s_mask_batch = torch.stack(s_mask)
    a_ids_batch = torch.stack(a_ids)
    a_mask_batch = torch.stack(a_mask)
    labs_batch = torch.stack(labs)

    # For sentence_texts, we just need the list of lists of strings
    s_texts_batch = list(s_texts_list)

    return (s_ids_batch, s_mask_batch, a_ids_batch, a_mask_batch, s_texts_batch, labs_batch)

