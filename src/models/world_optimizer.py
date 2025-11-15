"""
World Vector Optimization module.
Finds optimal z vector that minimizes constraint violation.
"""
import torch
import torch.nn as nn
from torch.optim import Adam


class WorldOptimizer:
    """
    Optimizes world vector z to minimize constraint violation E(A, z).
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        lr: float = 0.05,
        steps: int = 30,
        init_std: float = 0.01
    ):
        """
        Initialize world optimizer.
        
        Args:
            latent_dim: Dimension of world vector d
            lr: Learning rate for Adam optimizer
            steps: Number of optimization steps
            init_std: Standard deviation for initialization
        """
        self.latent_dim = latent_dim
        self.lr = lr
        self.steps = steps
        self.init_std = init_std
    
    def compute_energy(
        self,
        a_i: torch.Tensor,
        b_i: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy E(A, z) = sum_i (a_i^T z - b_i)^2
        
        Args:
            a_i: Constraint vectors (batch_size, num_sentences, latent_dim)
            b_i: Constraint scalars (batch_size, num_sentences)
            z: World vector (batch_size, latent_dim)
        
        Returns:
            Energy tensor (batch_size,)
        """
        # Compute a_i^T z for all sentences
        # z: (batch_size, latent_dim)
        # a_i: (batch_size, num_sentences, latent_dim)
        # Result: (batch_size, num_sentences)
        a_i_z = torch.bmm(
            a_i,
            z.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, num_sentences)
        
        # Compute (a_i^T z - b_i)^2
        diff = a_i_z - b_i  # (batch_size, num_sentences)
        squared_diff = diff ** 2
        
        # Sum over sentences
        energy = squared_diff.sum(dim=1)  # (batch_size,)
        
        return energy
    
    def optimize(
        self,
        a_i: torch.Tensor,
        b_i: torch.Tensor,
        device = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize world vector z for given constraints.
        
        Args:
            a_i: Constraint vectors (batch_size, num_sentences, latent_dim)
            b_i: Constraint scalars (batch_size, num_sentences)
            device: Device to run on
        
        Returns:
            tuple of (optimized_z, final_energy):
                - optimized_z: (batch_size, latent_dim)
                - final_energy: (batch_size,) - inconsistency score C(A)
        """
        batch_size, num_sentences, latent_dim = a_i.shape
        
        # Get device from input tensor if not provided
        if device is None:
            device = a_i.device
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Initialize z ~ N(0, 0.01)
        z = torch.randn(batch_size, latent_dim, device=device) * self.init_std
        z.requires_grad = True
        
        # Create optimizer for z
        optimizer = Adam([z], lr=self.lr)
        
        # Optimize
        for step in range(self.steps):
            optimizer.zero_grad()
            
            # Compute energy
            energy = self.compute_energy(a_i, b_i, z)
            
            # Backward pass
            loss = energy.sum()  # Sum over batch for optimization
            loss.backward()
            optimizer.step()
        
        # Final energy (inconsistency score)
        with torch.no_grad():
            final_energy = self.compute_energy(a_i, b_i, z)
        
        return z.detach(), final_energy

