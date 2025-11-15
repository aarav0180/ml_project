"""
Training module for BERT model.
"""
import torch
import torch.nn as nn


class Trainer:
    """
    Handles model training.
    """
    
    def __init__(self, model, optimizer, criterion, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train_epoch(self, train_dataloader):
        """
        Train for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            sent_id, mask, labels = batch
            
            # Move to device
            sent_id = sent_id.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            preds = self.model(sent_id, mask)
            loss = self.criterion(preds, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss / len(train_dataloader)

