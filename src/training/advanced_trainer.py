"""
Advanced trainer with three losses: classification, separation, and smoothness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedTrainer:
    """
    Trainer for advanced model with multiple loss components.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        device='cpu',
        lambda1=0.1,
        lambda2=0.1,
        alpha=0.001
    ):
        """
        Initialize advanced trainer.
        
        Args:
            model: Advanced model instance
            optimizer: Optimizer
            device: Device to train on
            lambda1: Weight for real news consistency loss
            lambda2: Weight for fake news consistency loss
            alpha: Weight for smoothness loss
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inconsistency_scores: torch.Tensor,
        a_i: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all three losses.
        
        Args:
            logits: Model predictions (batch_size, 2)
            labels: True labels (batch_size,)
            inconsistency_scores: C(A) scores (batch_size,)
            a_i: Constraint vectors (batch_size, num_sentences, latent_dim)
        
        Returns:
            Tuple of (total_loss, loss_dict)
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
        
        # 3. Constraint smoothness loss
        # L_smooth = alpha * sum_i ||a_i||^2
        a_i_norm_squared = torch.norm(a_i, dim=2) ** 2  # (batch_size, num_sentences)
        loss_smooth = self.alpha * a_i_norm_squared.sum()
        
        # Total loss
        total_loss = loss_cls + loss_sep + loss_smooth
        
        loss_dict = {
            'classification': loss_cls.item(),
            'separation': loss_sep.item(),
            'smoothness': loss_smooth.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_dataloader):
        """
        Train for one epoch.
        
        Args:
            train_dataloader: DataLoader for training
        
        Returns:
            Average loss dictionary
        """
        self.model.train()
        total_losses = {
            'classification': 0.0,
            'separation': 0.0,
            'smoothness': 0.0,
            'total': 0.0
        }
        num_batches = 0
        
        for batch in train_dataloader:
            # Unpack batch
            (sentence_ids, sentence_mask, article_ids, article_mask, labels) = batch
            
            # Move to device
            sentence_ids = sentence_ids.to(self.device)
            sentence_mask = sentence_mask.to(self.device)
            article_ids = article_ids.to(self.device)
            article_mask = article_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - need to get a_i for smoothness loss
            # We need to manually compute constraints
            sentence_embeddings = self.model.encode_sentences(sentence_ids, sentence_mask)
            a_i, b_i = self.model.constraint_generator(sentence_embeddings)
            
            # Optimize world vector
            if self.model._world_optimizer is None:
                from src.models.world_optimizer import WorldOptimizer
                self.model._world_optimizer = WorldOptimizer(
                    latent_dim=self.model.latent_dim,
                    lr=self.model.world_opt_lr,
                    steps=self.model.world_opt_steps
                )
            _, inconsistency_scores = self.model._world_optimizer.optimize(
                a_i, b_i, device=self.device
            )
            
            # Encode article and classify
            article_embedding = self.model.encode_article(article_ids, article_mask)
            inconsistency_expanded = inconsistency_scores.unsqueeze(1)
            combined_features = torch.cat([article_embedding, inconsistency_expanded], dim=1)
            logits = self.model.classifier(combined_features)
            
            # Compute losses
            loss, loss_dict = self.compute_losses(
                logits, labels, inconsistency_scores, a_i
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        return avg_losses

