"""Trainer for advanced model with plausibility-based consistency."""

from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Trainer for advanced model with plausibility-based consistency."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        lambda1: float = 0.1,
        lambda2: float = 0.1
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer for training
            device: Device to train on
            lambda1: Weight for real consistency loss
            lambda2: Weight for fake consistency loss
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.criterion = nn.CrossEntropyLoss()

    def compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inconsistency_scores: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute classification and separation losses.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            inconsistency_scores: Consistency scores from model
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_cls = self.criterion(logits, labels)

        # Consistency separation loss
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        if real_mask.sum() > 0:
            real_consistency = inconsistency_scores[real_mask].mean()
        else:
            real_consistency = torch.tensor(0.0, device=self.device)

        if fake_mask.sum() > 0:
            fake_consistency = inconsistency_scores[fake_mask].mean()
        else:
            fake_consistency = torch.tensor(0.0, device=self.device)

        loss_sep = self.lambda1 * real_consistency - self.lambda2 * fake_consistency
        total_loss = loss_cls + loss_sep

        loss_dict = {
            'classification': loss_cls.item(),
            'separation': loss_sep.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def train_epoch(self, train_dataloader, show_progress: bool = False) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_losses = {
            'classification': 0.0,
            'separation': 0.0,
            'total': 0.0
        }
        num_batches = 0

        if show_progress:
            pbar = tqdm(train_dataloader, desc="Training", unit="batch")
        else:
            pbar = train_dataloader

        for batch in pbar:
            (sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels) = batch

            article_ids = article_ids.to(self.device)
            article_mask = article_mask.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, inconsistency_scores = self.model(
                sentence_texts=sentence_texts,
                article_input_ids=article_ids,
                article_attention_mask=article_mask,
                return_inconsistency=True
            )

            loss, loss_dict = self.compute_losses(
                logits, labels, inconsistency_scores
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1

            if show_progress:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.4f}",
                    'Cls': f"{loss_dict['classification']:.4f}",
                    'Sep': f"{loss_dict['separation']:.4f}"
                })

        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        return avg_losses

