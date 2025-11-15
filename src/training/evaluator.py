"""
Evaluation module for BERT model.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """
    Handles model evaluation.
    """
    
    def __init__(self, model, criterion, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            criterion: Loss function
            device: Device to evaluate on
        """
        self.model = model
        self.criterion = criterion
        self.device = device
    
    def evaluate(self, val_dataloader):
        """
        Evaluate model on validation set.
        
        Args:
            val_dataloader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        for step, batch in enumerate(val_dataloader):
            sent_id, mask, labels = batch
            
            # Move to device
            sent_id = sent_id.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                preds = self.model(sent_id, mask)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_dataloader)
    
    def predict(self, test_seq, test_mask):
        """
        Make predictions on test data.
        
        Args:
            test_seq: Test input sequences
            test_mask: Test attention masks
        
        Returns:
            Predicted class labels
        """
        self.model.eval()
        
        test_seq = test_seq.to(self.device)
        test_mask = test_mask.to(self.device)
        
        with torch.no_grad():
            preds = self.model(test_seq, test_mask).cpu().numpy()
        
        return np.argmax(preds, axis=1)
    
    def evaluate_test(self, test_seq, test_mask, test_labels):
        """
        Evaluate model on test set and print classification report.
        
        Args:
            test_seq: Test input sequences
            test_mask: Test attention masks
            test_labels: True test labels
        
        Returns:
            Classification report as string
        """
        preds = self.predict(test_seq, test_mask)
        
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.cpu().numpy()
        
        report = classification_report(test_labels, preds)
        return report, preds

