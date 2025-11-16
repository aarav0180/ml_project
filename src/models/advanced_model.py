"""
Advanced Fake News Detection Model with Plausibility-based Consistency Scoring.
Combines BERT article encoding with plausibility-based inconsistency score C(A).
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from .plausibility_scorer import PlausibilityWorldModel


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
        Initialize advanced model.
        
        Args:
            bert_model_name: BERT model name
            freeze_bert: Whether to freeze BERT parameters
            plausibility_model_path: Path to plausibility model directory
        """
        super(AdvancedFakeNewsModel, self).__init__()
        
        # Article encoder (BERT)
        self.article_bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.article_bert.parameters():
                param.requires_grad = False
        
        # Plausibility scorer (frozen, used for inference)
        self.plausibility_model = PlausibilityWorldModel(
            model_path=plausibility_model_path,
            device=None  # Will be set when model moves to device
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
            article_input_ids: (batch_size, max_seq_len)
            article_attention_mask: (batch_size, max_seq_len)
        
        Returns:
            Article embedding (batch_size, 768)
        """
        # Since BERT is frozen, use no_grad for efficiency
        with torch.no_grad():
            outputs = self.article_bert(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            )
        
        # Get CLS embedding
        article_embedding = outputs.pooler_output  # (batch_size, 768)
        # Re-enable gradients for classifier
        article_embedding = article_embedding.requires_grad_(True)
        
        return article_embedding
    
    def compute_consistency_score(
        self,
        plausibility_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute consistency score C(A) from plausibility scores.
        
        Args:
            plausibility_scores: (batch_size, num_sentences) plausibility scores [0,1]
        
        Returns:
            tuple of (C(A), mean_plausibility, max_violation):
                - C(A): (batch_size,) consistency score (energy-like)
                - mean_plausibility: (batch_size,) mean plausibility
                - max_violation: (batch_size,) maximum violation score
        """
        # Compute violations: 1 - plausibility
        violations = 1.0 - plausibility_scores  # (batch_size, num_sentences)
        
        # Consistency score: mean of squared violations (energy-like formula)
        consistency_scores = (violations ** 2).mean(dim=1)  # (batch_size,)
        
        # Mean plausibility
        mean_plausibility = plausibility_scores.mean(dim=1)  # (batch_size,)
        
        # Max violation (worst case)
        max_violation = violations.max(dim=1)[0]  # (batch_size,)
        
        return consistency_scores, mean_plausibility, max_violation
    
    def forward(
        self,
        sentence_texts: list[list[str]],
        article_input_ids: torch.Tensor,
        article_attention_mask: torch.Tensor,
        return_inconsistency: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sentence_texts: List of lists of sentence strings (batch_size, num_sentences)
            article_input_ids: (batch_size, max_seq_len)
            article_attention_mask: (batch_size, max_seq_len)
            return_inconsistency: If True, also return inconsistency scores
        
        Returns:
            tuple of (logits, inconsistency_scores):
                - logits: (batch_size, 2)
                - inconsistency_scores: (batch_size,) if return_inconsistency else None
        """
        batch_size = len(sentence_texts)
        
        # 1. Score sentences for plausibility
        # Process each article's sentences
        all_plausibility_scores = []
        
        for article_sentences in sentence_texts:
            # Score all sentences in this article
            article_scores = self.plausibility_model.score_batch(article_sentences)
            all_plausibility_scores.append(article_scores)
        
        # Convert to tensor
        # Find max number of sentences (for padding)
        max_sentences = max(len(scores) for scores in all_plausibility_scores)
        
        # Pad to same length
        plausibility_tensor = torch.zeros(batch_size, max_sentences, device=article_input_ids.device)
        for i, scores in enumerate(all_plausibility_scores):
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
        
        # 4. Combine features: concat([x_A, C(A), mean_plausibility, max_violation])
        consistency_expanded = consistency_scores.unsqueeze(1)  # (batch_size, 1)
        mean_plausibility_expanded = mean_plausibility.unsqueeze(1)  # (batch_size, 1)
        max_violation_expanded = max_violation.unsqueeze(1)  # (batch_size, 1)
        
        combined_features = torch.cat(
            [
                article_embedding,  # (batch_size, 768)
                consistency_expanded,  # (batch_size, 1)
                mean_plausibility_expanded,  # (batch_size, 1)
                max_violation_expanded  # (batch_size, 1)
            ],
            dim=1
        )  # (batch_size, 771)
        
        # 5. Classify
        logits = self.classifier(combined_features)  # (batch_size, 2)
        
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
