"""
Constraint Generator MLP for generating constraints (a_i, b_i) from sentence embeddings.
"""
import torch
import torch.nn as nn


class ConstraintGenerator(nn.Module):
    """
    MLP that generates constraints (a_i, b_i) from BERT sentence embeddings.
    
    Input: 768-d BERT CLS embedding
    Output: a_i (d-dim vector) and b_i (scalar)
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, latent_dim=32):
        """
        Initialize constraint generator.
        
        Args:
            input_dim: Dimension of input BERT embedding (768)
            hidden_dim: Hidden layer dimension (256)
            latent_dim: Dimension of world vector d (32)
        """
        super(ConstraintGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # MLP: 768 -> 256 -> (d + 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim + 1)  # +1 for b_i
    
    def forward(self, sentence_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate constraints from sentence embeddings.
        
        Args:
            sentence_embeddings: Tensor of shape (batch_size, num_sentences, 768)
        
        Returns:
            tuple of (a_i, b_i):
                - a_i: Tensor of shape (batch_size, num_sentences, latent_dim)
                - b_i: Tensor of shape (batch_size, num_sentences)
        """
        batch_size, num_sentences, _ = sentence_embeddings.shape
        
        # Reshape to (batch_size * num_sentences, 768)
        x = sentence_embeddings.view(-1, self.input_dim)
        
        # Forward through MLP
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # (batch_size * num_sentences, latent_dim + 1)
        
        # Split into a_i and b_i
        a_i = x[:, :self.latent_dim]  # (batch_size * num_sentences, latent_dim)
        b_i = x[:, self.latent_dim]   # (batch_size * num_sentences)
        
        # Normalize a_i: a_i = a_i / ||a_i||
        a_i_norm = torch.norm(a_i, dim=1, keepdim=True)
        a_i_norm = torch.clamp(a_i_norm, min=1e-8)  # Avoid division by zero
        a_i = a_i / a_i_norm
        
        # Reshape back
        a_i = a_i.view(batch_size, num_sentences, self.latent_dim)
        b_i = b_i.view(batch_size, num_sentences)
        
        return a_i, b_i

