"""
Plausibility Scorer using trained DistilBERT model.
Scores sentences for plausibility (0-1 range).
"""
import os
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import List, Union


class PlausibilityWorldModel:
    """
    Wrapper for trained DistilBERT plausibility model.
    Scores claims/sentences for plausibility.
    """
    
    def __init__(self, model_path='plausability_model_final', device=None):
        """
        Initialize plausibility model.
        
        Args:
            model_path: Path to directory containing model files
            device: Device to load model on ('cuda' or 'cpu'). Auto-detected if None
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_path = model_path
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Plausibility model not found at {model_path}. "
                f"Please ensure the model files are in the correct location."
            )
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Cache for sentence scores (optional optimization)
        self._cache = {}
    
    def score_claim(self, sentence: str) -> float:
        """
        Score a single sentence for plausibility.
        
        Args:
            sentence: Input sentence string
        
        Returns:
            Plausibility score [0, 1] where 1 = plausible, 0 = implausible
        """
        # Check cache
        if sentence in self._cache:
            return self._cache[sentence]
        
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Score
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Regression output: single value
            score = outputs.logits.squeeze().item()
        
        # Normalize to [0, 1] if needed (assuming model outputs in this range)
        # Clamp to ensure valid range
        score = max(0.0, min(1.0, score))
        
        # Cache result
        self._cache[sentence] = score
        
        return score
    
    def score_batch(self, sentences: List[str]) -> List[float]:
        """
        Score multiple sentences in batch for efficiency.
        
        Args:
            sentences: List of sentence strings
        
        Returns:
            List of plausibility scores [0, 1]
        """
        if not sentences:
            return []
        
        # Filter out cached sentences
        uncached_sentences = []
        uncached_indices = []
        scores = [None] * len(sentences)
        
        for i, sent in enumerate(sentences):
            if sent in self._cache:
                scores[i] = self._cache[sent]
            else:
                uncached_sentences.append(sent)
                uncached_indices.append(i)
        
        # Batch process uncached sentences
        if uncached_sentences:
            # Tokenize batch
            inputs = self.tokenizer(
                uncached_sentences,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Score batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze()
                
                # Handle single vs batch output
                if batch_scores.dim() == 0:
                    batch_scores = batch_scores.unsqueeze(0)
                
                batch_scores = batch_scores.cpu().numpy()
            
            # Normalize and cache
            for idx, score in zip(uncached_indices, batch_scores):
                score = max(0.0, min(1.0, float(score)))
                scores[idx] = score
                self._cache[sentences[idx]] = score
        
        return scores
    
    def clear_cache(self):
        """Clear the score cache."""
        self._cache.clear()
    
    def compute_violation_score(self, plausibility: float) -> float:
        """
        Convert plausibility to violation score.
        
        Args:
            plausibility: Plausibility score [0, 1]
        
        Returns:
            Violation score [0, 1] where 1 = high violation
        """
        return 1.0 - plausibility
    
    def is_uncertain(self, plausibility: float, threshold: float = 0.5) -> bool:
        """
        Check if plausibility score indicates uncertainty/implausibility.
        
        Args:
            plausibility: Plausibility score [0, 1]
            threshold: Threshold below which is considered uncertain
        
        Returns:
            True if uncertain/implausible
        """
        return plausibility < threshold

