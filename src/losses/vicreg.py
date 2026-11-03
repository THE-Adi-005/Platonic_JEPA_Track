"""
VICReg Loss for Joint-Embedding Predictive Architectures (JEPAs)
Implements:
1. Invariance Loss (MSE Prediction Error)
2. Variance Regularization (Hinge on standard deviation, prevents collapse)
3. Covariance Regularization (Decorrelation, prevents redundant representations)
Reference: Bardes, Ponce, LeCun (ICLR 2022) / Balestriero & LeCun (arXiv:2610.39302)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VICRegLoss(nn.Module):
    def __init__(
        self,
        lambda_inv: float = 25.0,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: projected video embedding z_v' (B, d)
        y: target attribute embedding z_a (B, d)
        """
        batch_size, dim = x.size()

        # 1. Invariance Loss (Prediction Error)
        sim_loss = F.mse_loss(x, y)

        # 2. Variance Loss (Anti-collapse hinge)
        # Center embeddings
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)

        std_x = torch.sqrt(x_centered.var(dim=0) + self.eps)
        std_y = torch.sqrt(y_centered.var(dim=0) + self.eps)

        var_loss = (
            torch.mean(F.relu(self.gamma - std_x)) +
            torch.mean(F.relu(self.gamma - std_y))
        ) / 2.0

        # 3. Covariance Loss (Decorrelation)
        cov_x = (x_centered.T @ x_centered) / (batch_size - 1)
        cov_y = (y_centered.T @ y_centered) / (batch_size - 1)

        # Off-diagonal elements
        cov_x_off = cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()
        cov_y_off = cov_y.pow(2).sum() - cov_y.diag().pow(2).sum()

        cov_loss = (cov_x_off / dim + cov_y_off / dim) / 2.0

        # Total Loss
        total_loss = (
            self.lambda_inv * sim_loss +
            self.mu_var * var_loss +
            self.nu_cov * cov_loss
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_invariance": sim_loss.item(),
            "loss_variance": var_loss.item(),
            "loss_covariance": cov_loss.item()
        }

        return total_loss, metrics
