"""
Advanced trainer with classification and separation losses.
Uses plausibility-based consistency scoring.
"""
import torch
import torch.nn as nn
from tqdm import tqdm


class AdvancedTrainer:
    """
    Trainer for advanced model with plausibility-based consistency.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        device='cpu',
        lambda1=0.1,
        lambda2=0.1
    ):
        """
        Initialize advanced trainer.
        
        Args:
            model: Advanced model instance
            optimizer: Optimizer
            device: Device to train on
            lambda1: Weight for real news consistency loss
            lambda2: Weight for fake news consistency loss
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
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute classification and separation losses.
        
        Args:
            logits: Model predictions (batch_size, 2)
            labels: True labels (batch_size,)
            inconsistency_scores: C(A) scores (batch_size,)
        
        Returns:
            tuple of (total_loss, loss_dict)
        """
        # 1. Classification loss
        loss_cls = self.criterion(logits, labels)
        
        # 2. Consistency separation loss
        # Real news (label=0) should have low C(A)
        # Fake news (label=1) should have high C(A)
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
        
        # L_sep = lambda1 * E[C(A)]_real - lambda2 * E[C(A)]_fake
        loss_sep = self.lambda1 * real_consistency - self.lambda2 * fake_consistency
        
        # Total loss (no smoothness loss)
        total_loss = loss_cls + loss_sep
        
        loss_dict = {
            'classification': loss_cls.item(),
            'separation': loss_sep.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_dataloader, show_progress=False):
        """
        Train for one epoch.
        
        Args:
            train_dataloader: DataLoader for training
            show_progress: If True, show progress bar
        
        Returns:
            Average loss dictionary
        """
        self.model.train()
        total_losses = {
            'classification': 0.0,
            'separation': 0.0,
            'total': 0.0
        }
        num_batches = 0
        
        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(train_dataloader, desc="Training", unit="batch")
        else:
            pbar = train_dataloader
        
        for batch in pbar:
            # Unpack batch
            (sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels) = batch
            
            # Move to device
            article_ids = article_ids.to(self.device)
            article_mask = article_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with plausibility scoring
            logits, inconsistency_scores = self.model(
                sentence_texts=sentence_texts,
                article_input_ids=article_ids,
                article_attention_mask=article_mask,
                return_inconsistency=True
            )
            
            # Compute losses
            loss, loss_dict = self.compute_losses(
                logits, labels, inconsistency_scores
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
            
            # Update progress bar
            if show_progress:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.4f}",
                    'Cls': f"{loss_dict['classification']:.4f}",
                    'Sep': f"{loss_dict['separation']:.4f}"
                })
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        return avg_losses
