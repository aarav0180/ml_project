"""
Advanced evaluator for the advanced model.
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score


class AdvancedEvaluator:
    """
    Evaluator for advanced model with inconsistency score analysis.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Advanced model instance
            device: Device to evaluate on
        """
        self.model = model
        self.device = device
    
    def evaluate(self, val_dataloader):
        """
        Evaluate model on validation set.
        
        Args:
            val_dataloader: DataLoader for validation data
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_inconsistency_scores = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                (sentence_ids, sentence_mask, article_ids, article_mask, labels) = batch
                
                # Move to device
                sentence_ids = sentence_ids.to(self.device)
                sentence_mask = sentence_mask.to(self.device)
                article_ids = article_ids.to(self.device)
                article_mask = article_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, inconsistency_scores = self.model(
                    sentence_ids, sentence_mask,
                    article_ids, article_mask,
                    return_inconsistency=True
                )
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_inconsistency_scores.append(inconsistency_scores.cpu().numpy())
        
        # Concatenate all results
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_inconsistency_scores = np.concatenate(all_inconsistency_scores, axis=0)
        
        # Get predictions
        predictions = np.argmax(all_logits, axis=1)
        probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions)
        roc_auc = roc_auc_score(all_labels, probabilities[:, 1])
        
        # Inconsistency score analysis
        real_scores = all_inconsistency_scores[all_labels == 0]
        fake_scores = all_inconsistency_scores[all_labels == 1]
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mean_inconsistency_real': float(real_scores.mean()) if len(real_scores) > 0 else 0.0,
            'mean_inconsistency_fake': float(fake_scores.mean()) if len(fake_scores) > 0 else 0.0,
            'std_inconsistency_real': float(real_scores.std()) if len(real_scores) > 0 else 0.0,
            'std_inconsistency_fake': float(fake_scores.std()) if len(fake_scores) > 0 else 0.0
        }
        
        # Classification report
        report = classification_report(all_labels, predictions)
        
        return metrics, report, {
            'predictions': predictions,
            'labels': all_labels,
            'inconsistency_scores': all_inconsistency_scores,
            'probabilities': probabilities
        }

