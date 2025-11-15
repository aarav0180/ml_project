"""
Advanced Fake News Detection Model with Constraint-based Consistency Scoring.
Combines BERT article encoding with inconsistency score C(A).
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple

from .constraint_generator import ConstraintGenerator
from .world_optimizer import WorldOptimizer


class AdvancedFakeNewsModel(nn.Module):
    """
    Advanced model that uses:
    1. Sentence-level BERT encoding with constraint generation
    2. World vector optimization for inconsistency scoring
    3. Full article BERT encoding
    4. Combined classifier using both features
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        latent_dim: int = 32,
        hidden_dim: int = 256,
        freeze_bert: bool = True,
        world_opt_lr: float = 0.05,
        world_opt_steps: int = 30
    ):
        """
        Initialize advanced model.
        
        Args:
            bert_model_name: BERT model name
            latent_dim: Dimension of world vector d
            hidden_dim: Hidden dimension for constraint generator
            freeze_bert: Whether to freeze BERT parameters
            world_opt_lr: Learning rate for world vector optimization
            world_opt_steps: Number of optimization steps for world vector
        """
        super(AdvancedFakeNewsModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Sentence encoder (BERT)
        self.sentence_bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.sentence_bert.parameters():
                param.requires_grad = False
        
        # Article encoder (BERT) - separate instance
        self.article_bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.article_bert.parameters():
                param.requires_grad = False
        
        # Constraint generator
        self.constraint_generator = ConstraintGenerator(
            input_dim=768,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # World optimizer (not a module, just a helper class)
        self.world_opt_lr = world_opt_lr
        self.world_opt_steps = world_opt_steps
        self._world_optimizer = None
        
        # Combined classifier
        # Input: article embedding (768) + inconsistency score (1) = 769
        self.classifier = nn.Sequential(
            nn.Linear(769, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def encode_sentences(
        self,
        sentence_input_ids: torch.Tensor,
        sentence_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode sentences using BERT.
        
        Args:
            sentence_input_ids: (batch_size, num_sentences, max_seq_len)
            sentence_attention_mask: (batch_size, num_sentences, max_seq_len)
        
        Returns:
            Sentence embeddings (batch_size, num_sentences, 768)
        """
        batch_size, num_sentences, seq_len = sentence_input_ids.shape
        
        # Reshape to (batch_size * num_sentences, seq_len)
        input_ids = sentence_input_ids.view(-1, seq_len)
        attention_mask = sentence_attention_mask.view(-1, seq_len)
        
        # Encode with BERT
        with torch.no_grad():
            outputs = self.sentence_bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get CLS embeddings
        cls_embeddings = outputs.pooler_output  # (batch_size * num_sentences, 768)
        
        # Reshape back
        cls_embeddings = cls_embeddings.view(batch_size, num_sentences, 768)
        
        return cls_embeddings
    
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
        with torch.no_grad():
            outputs = self.article_bert(
                input_ids=article_input_ids,
                attention_mask=article_attention_mask
            )
        
        # Get CLS embedding
        article_embedding = outputs.pooler_output  # (batch_size, 768)
        
        return article_embedding
    
    def forward(
        self,
        sentence_input_ids: torch.Tensor,
        sentence_attention_mask: torch.Tensor,
        article_input_ids: torch.Tensor,
        article_attention_mask: torch.Tensor,
        return_inconsistency: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sentence_input_ids: (batch_size, num_sentences, max_seq_len)
            sentence_attention_mask: (batch_size, num_sentences, max_seq_len)
            article_input_ids: (batch_size, max_seq_len)
            article_attention_mask: (batch_size, max_seq_len)
            return_inconsistency: If True, also return inconsistency scores
        
        Returns:
            Tuple of (logits, inconsistency_scores):
                - logits: (batch_size, 2)
                - inconsistency_scores: (batch_size,) if return_inconsistency else None
        """
        # 1. Encode sentences
        sentence_embeddings = self.encode_sentences(
            sentence_input_ids,
            sentence_attention_mask
        )
        
        # 2. Generate constraints
        a_i, b_i = self.constraint_generator(sentence_embeddings)
        
        # 3. Optimize world vector and get inconsistency score
        if self._world_optimizer is None:
            self._world_optimizer = WorldOptimizer(
                latent_dim=self.latent_dim,
                lr=self.world_opt_lr,
                steps=self.world_opt_steps
            )
        _, inconsistency_scores = self._world_optimizer.optimize(
            a_i, b_i, device=sentence_input_ids.device
        )
        
        # 4. Encode full article
        article_embedding = self.encode_article(
            article_input_ids,
            article_attention_mask
        )
        
        # 5. Combine features: concat([x_A, C(A)])
        inconsistency_scores_expanded = inconsistency_scores.unsqueeze(1)  # (batch_size, 1)
        combined_features = torch.cat(
            [article_embedding, inconsistency_scores_expanded],
            dim=1
        )  # (batch_size, 769)
        
        # 6. Classify
        logits = self.classifier(combined_features)  # (batch_size, 2)
        
        if return_inconsistency:
            return logits, inconsistency_scores
        return logits, None

