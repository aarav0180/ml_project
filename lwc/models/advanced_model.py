"""Advanced Fake News Model combining BERT and Latent World Consistency."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModel
from lwc.models.plausibility import PlausibilityWorldModel
import logging

logger = logging.getLogger(__name__)


class AdvancedFakeNewsModel(nn.Module):
    """
    Advanced model that uses:
    1. Sentence-level plausibility scoring
    2. Consistency score C(A) from plausibility violations
    3. Full article BERT encoding
    4. Combined classifier using both features
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = True,
        plausibility_model_path: str = 'plausability_model_final'
    ):
        """
        Initialize the advanced fake news model.
        
        Args:
            bert_model_name: Name of the BERT model to use
            freeze_bert: Whether to freeze BERT parameters
            plausibility_model_path: Path to the plausibility model
        """
        super(AdvancedFakeNewsModel, self).__init__()

        # Article encoder (BERT)
        logger.info(f"Loading BERT model: {bert_model_name}")
        self.article_bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.article_bert.parameters():
                param.requires_grad = False
            logger.info("BERT parameters frozen")

        # Plausibility scorer (frozen, used for inference)
        self.plausibility_model = PlausibilityWorldModel(
            model_path=plausibility_model_path,
            device=None
        )

        # Combined classifier
        # Input: article embedding (768) + C(A) (1) + mean_plausibility (1) + max_violation (1) = 771
        self.classifier = nn.Sequential(
            nn.Linear(771, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def encode_article(
        self,
        article_input_ids: torch.Tensor,
        article_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode full article using BERT.
        
        Args:
            article_input_ids: Token IDs for the article
            article_attention_mask: Attention mask for the article
            
        Returns:
            Article embedding tensor
        """
        with torch.no_grad():
            outputs = self.article_bert(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            )

        article_embedding = outputs.pooler_output
        article_embedding = article_embedding.requires_grad_(True)
        return article_embedding

    def compute_consistency_score(
        self,
        plausibility_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute consistency score C(A) from plausibility scores.
        
        Args:
            plausibility_scores: Tensor of plausibility scores
            
        Returns:
            Tuple of (consistency_scores, mean_plausibility, max_violation)
        """
        violations = 1.0 - plausibility_scores
        consistency_scores = (violations ** 2).mean(dim=1)
        mean_plausibility = plausibility_scores.mean(dim=1)
        max_violation = violations.max(dim=1)[0]

        return consistency_scores, mean_plausibility, max_violation

    def forward(
        self,
        sentence_texts: List[List[str]],
        article_input_ids: torch.Tensor,
        article_attention_mask: torch.Tensor,
        return_inconsistency: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            sentence_texts: List of lists of sentence strings for each article
            article_input_ids: Token IDs for full articles
            article_attention_mask: Attention mask for full articles
            return_inconsistency: Whether to return inconsistency scores
            
        Returns:
            Tuple of (logits, inconsistency_scores or None)
        """
        batch_size = article_input_ids.shape[0]

        # 1. Score sentences for plausibility
        all_plausibility_scores = []
        for article_sentences in sentence_texts:
            article_scores = self.plausibility_model.score_batch(article_sentences)
            all_plausibility_scores.append(article_scores)

        # Convert to tensor
        max_sentences = max(len(scores) for scores in all_plausibility_scores) if all_plausibility_scores else 0
        if max_sentences == 0:
            max_sentences = 1  # Minimal valid size to avoid errors

        plausibility_tensor = torch.zeros(batch_size, max_sentences, device=article_input_ids.device)
        for i, scores in enumerate(all_plausibility_scores):
            if scores:
                plausibility_tensor[i, :len(scores)] = torch.tensor(
                    scores, device=article_input_ids.device
                )

        # 2. Compute consistency score C(A)
        consistency_scores, mean_plausibility, max_violation = self.compute_consistency_score(
            plausibility_tensor
        )

        # 3. Encode full article
        article_embedding = self.encode_article(
            article_input_ids,
            article_attention_mask
        )

        # 4. Combine features
        combined_features = torch.cat(
            [
                article_embedding,
                consistency_scores.unsqueeze(1),
                mean_plausibility.unsqueeze(1),
                max_violation.unsqueeze(1)
            ],
            dim=1
        )

        # 5. Classify
        logits = self.classifier(combined_features)

        if return_inconsistency:
            return logits, consistency_scores
        return logits, None

    def to(self, device):
        """Move model to device and update plausibility model device."""
        super().to(device)
        if hasattr(self, 'plausibility_model'):
            self.plausibility_model.device = device
            self.plausibility_model.model.to(device)
        return self

