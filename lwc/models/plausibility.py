"""Plausibility World Model for sentence-level plausibility scoring."""

import os
from typing import List
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import logging

logger = logging.getLogger(__name__)


class PlausibilityWorldModel:
    """Wrapper for trained DistilBERT plausibility model."""

    def __init__(self, model_path: str = 'plausability_model_final', device=None):
        """
        Initialize the plausibility model.
        
        Args:
            model_path: Path to the pre-trained plausibility model directory
            device: Device to load the model on (None for auto-detection)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Plausibility model not found at {model_path}."
            )

        logger.info(f"Loading plausibility model from {model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self._cache = {}

    def score_claim(self, sentence: str) -> float:
        """
        Score a single sentence for plausibility.
        
        Args:
            sentence: The sentence to score
            
        Returns:
            Plausibility score between 0.0 and 1.0
        """
        if sentence in self._cache:
            return self._cache[sentence]

        inputs = self.tokenizer(
            sentence,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()

        score = max(0.0, min(1.0, score))
        self._cache[sentence] = score
        return score

    def score_batch(self, sentences: List[str]) -> List[float]:
        """
        Score multiple sentences in batch for efficiency.
        
        Args:
            sentences: List of sentences to score
            
        Returns:
            List of plausibility scores
        """
        if not sentences:
            return []

        uncached_sentences = []
        uncached_indices = []
        scores = [None] * len(sentences)

        for i, sent in enumerate(sentences):
            if sent in self._cache:
                scores[i] = self._cache[sent]
            else:
                uncached_sentences.append(sent)
                uncached_indices.append(i)

        if uncached_sentences:
            inputs = self.tokenizer(
                uncached_sentences,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze()

                if batch_scores.dim() == 0:
                    batch_scores = batch_scores.unsqueeze(0)

                batch_scores = batch_scores.cpu().numpy()

            for idx, score in zip(uncached_indices, batch_scores):
                score = max(0.0, min(1.0, float(score)))
                scores[idx] = score
                self._cache[sentences[idx]] = score

        return scores

