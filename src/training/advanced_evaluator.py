"""
Advanced evaluator for the advanced model with plausibility statistics.
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score


class AdvancedEvaluator:
    """
    Evaluator for advanced model with inconsistency score and plausibility analysis.
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
            Dictionary with metrics, classification report, and detailed results
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_inconsistency_scores = []
        all_plausibility_scores = []  # Track plausibility scores
        
        with torch.no_grad():
            for batch in val_dataloader:
                (sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels) = batch
                
                # Move to device
                article_ids = article_ids.to(self.device)
                article_mask = article_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, inconsistency_scores = self.model(
                    sentence_texts=sentence_texts,
                    article_input_ids=article_ids,
                    article_attention_mask=article_mask,
                    return_inconsistency=True
                )
                
                # Collect plausibility scores from model
                # Score all sentences in batch
                batch_plausibility = []
                for article_sentences in sentence_texts:
                    article_plausibility = self.model.plausibility_model.score_batch(article_sentences)
                    batch_plausibility.append(article_plausibility)
                all_plausibility_scores.extend(batch_plausibility)
                
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
        
        # Plausibility statistics
        # Flatten plausibility scores and compute statistics
        flat_plausibility = [p for article_scores in all_plausibility_scores for p in article_scores]
        if flat_plausibility:
            mean_plausibility = np.mean(flat_plausibility)
            std_plausibility = np.std(flat_plausibility)
            min_plausibility = np.min(flat_plausibility)
            max_plausibility = np.max(flat_plausibility)
            
            # Count uncertain/implausible sentences (< 0.5)
            uncertain_count = sum(1 for p in flat_plausibility if p < 0.5)
            uncertain_ratio = uncertain_count / len(flat_plausibility) if flat_plausibility else 0.0
        else:
            mean_plausibility = std_plausibility = min_plausibility = max_plausibility = 0.0
            uncertain_ratio = 0.0
        
        # Per-article mean plausibility
        article_mean_plausibility = [np.mean(scores) if scores else 0.0 for scores in all_plausibility_scores]
        real_plausibility = [article_mean_plausibility[i] for i in range(len(article_mean_plausibility)) if all_labels[i] == 0]
        fake_plausibility = [article_mean_plausibility[i] for i in range(len(article_mean_plausibility)) if all_labels[i] == 1]
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mean_inconsistency_real': float(real_scores.mean()) if len(real_scores) > 0 else 0.0,
            'mean_inconsistency_fake': float(fake_scores.mean()) if len(fake_scores) > 0 else 0.0,
            'std_inconsistency_real': float(real_scores.std()) if len(real_scores) > 0 else 0.0,
            'std_inconsistency_fake': float(fake_scores.std()) if len(fake_scores) > 0 else 0.0,
            # Plausibility statistics
            'mean_plausibility': float(mean_plausibility),
            'std_plausibility': float(std_plausibility),
            'min_plausibility': float(min_plausibility),
            'max_plausibility': float(max_plausibility),
            'uncertain_ratio': float(uncertain_ratio),
            'mean_plausibility_real': float(np.mean(real_plausibility)) if len(real_plausibility) > 0 else 0.0,
            'mean_plausibility_fake': float(np.mean(fake_plausibility)) if len(fake_plausibility) > 0 else 0.0,
        }
        
        # Classification report
        report = classification_report(all_labels, predictions)
        
        return metrics, report, {
            'predictions': predictions,
            'labels': all_labels,
            'inconsistency_scores': all_inconsistency_scores,
            'probabilities': probabilities,
            'plausibility_scores': all_plausibility_scores
        }
