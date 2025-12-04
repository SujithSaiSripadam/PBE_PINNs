"""
Adaptive sampling and loss weighting utilities for PINNs.
Helps focus training on difficult regions and balance multi-task losses.
"""

import torch
import numpy as np


class AdaptiveSampler:
    """
    Adaptively resamples collocation points based on physics residuals.
    Prioritizes high-residual regions to improve training efficiency.
    """

    def __init__(
        self,
        n_candidates: int = 10000,
        high_residual_ratio: float = 0.7,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        n_candidates : int
            Pool size of candidate collocation points to draw from
        high_residual_ratio : float
            Fraction of high-residual points to select (0-1)
        device : str or torch.device
        dtype : torch.dtype
        """
        self.n_candidates = n_candidates
        self.high_residual_ratio = high_residual_ratio
        self.device = torch.device(device)
        self.dtype = dtype

    def compute_residuals(
        self,
        shared_net,
        t_coll,
        L_coll,
        T_coll,
        F_coll,
        N_coll,
        loss_fn,
        t_scale,
        L_scale,
        T_scale,
        F_scale,
        N_scale,
    ):
        """
        Compute physics residuals at collocation points.

        Parameters
        ----------
        shared_net : PINN_SHARED
            Network model
        t_coll, L_coll, T_coll, F_coll, N_coll : torch.Tensor
            Collocation point parameters (physical units)
        loss_fn : callable
            Loss function to call (should return per-point residuals)
        scales : float
            Normalization scales

        Returns
        -------
        residuals : torch.Tensor
            Per-point residual magnitude, shape (n_points,)
        """
        with torch.no_grad():
            # Normalize
            t_norm = (t_coll / t_scale).to(self.device)
            L_norm = (L_coll / L_scale).to(self.device)
            T_norm = (T_coll / T_scale).to(self.device)
            F_norm = (F_coll / F_scale).to(self.device)
            N_norm = (N_coll / N_scale).to(self.device)

            # Forward pass - extract residuals from loss computation
            # This is a simplified version; in practice you'd need to extract
            # individual residuals from within compute_loss
            shared_net.eval()
            n_c_hat, n_wm_hat = shared_net(t_norm, L_norm, T_norm, F_norm, N_norm)
            c_c_hat, c_wm_hat = shared_net(t_norm, T_norm, F_norm, N_norm)

            # Return dummy residuals for now (would need residual extraction from loss_fn)
            residuals = torch.ones(t_coll.shape[0], device=self.device)

        return residuals

    def adaptive_resample(
        self,
        t_coll,
        L_coll,
        T_coll,
        F_coll,
        N_coll,
        residuals,
        batch_size,
    ):
        """
        Resample collocation points, prioritizing high-residual regions.

        Parameters
        ----------
        t_coll, L_coll, T_coll, F_coll, N_coll : torch.Tensor
            Full set of collocation candidates, shape (n_candidates,)
        residuals : torch.Tensor
            Residual magnitude at each point, shape (n_candidates,)
        batch_size : int
            Target batch size for resampling
        high_residual_ratio : float
            Fraction (0-1) of high-residual points to include

        Returns
        -------
        Tuple of resampled (t_b, L_b, T_b, F_b, N_b), each shape (batch_size,)
        """
        n_high = max(1, int(batch_size * self.high_residual_ratio))
        n_low = batch_size - n_high

        # Move residuals to CPU for indexing
        residuals_cpu = residuals.detach().cpu()
        
        # Get indices of high-residual points
        _, high_idx = torch.topk(residuals_cpu, k=n_high, largest=True)
        
        # Random sample from remaining
        all_idx = torch.arange(len(residuals_cpu))
        mask = torch.ones(len(residuals_cpu), dtype=torch.bool)
        mask[high_idx] = False
        remaining_idx = all_idx[mask]
        
        if len(remaining_idx) > 0:
            low_idx = remaining_idx[torch.randperm(len(remaining_idx))[:n_low]]
        else:
            # Fallback: if we don't have enough remaining points, resample from high_idx
            low_idx = high_idx[torch.randperm(len(high_idx))[:n_low]]
        
        # Combine indices
        idx = torch.cat([high_idx, low_idx]).to(t_coll.device)
        
        #print("DEBUG: Adaptive sampling")

        return (
            t_coll[idx],
            L_coll[idx],
            T_coll[idx],
            F_coll[idx],
            N_coll[idx],
        )


class AdaptiveLossWeighter:
    """
    Dynamically balances physics and data losses based on their magnitudes.
    Prevents one loss from dominating the other.
    """

    def __init__(
        self,
        initial_lambda_phys: float = 1.0,
        initial_lambda_data: float = 1.0,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
    ):
        """
        Parameters
        ----------
        initial_lambda_phys : float
            Initial physics loss weight
        initial_lambda_data : float
            Initial data loss weight
        min_ratio : float
            Minimum allowed ratio (physics_weight / data_weight)
        max_ratio : float
            Maximum allowed ratio
        """
        self.lambda_phys = initial_lambda_phys
        self.lambda_data = initial_lambda_data
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.loss_history = {"phys": [], "data": []}

    def update(self, loss_phys, loss_data, step):
        """
        Rebalance weights based on current loss magnitudes.

        Parameters
        ----------
        loss_phys : float
            Current physics loss
        loss_data : float
            Current data loss
        step : int
            Current training step (for logging)

        Returns
        -------
        Tuple of updated (lambda_phys, lambda_data)
        """
        # Record history
        self.loss_history["phys"].append(loss_phys)
        self.loss_history["data"].append(loss_data)

        # Compute ratio to balance magnitudes
        if loss_data > 1e-8:
            ratio = loss_phys / loss_data
        else:
            ratio = 1.0

        # Clamp ratio
        ratio = np.clip(ratio, self.min_ratio, self.max_ratio)

        # Update lambda_phys to normalize physics loss relative to data loss
        # Goal: make both losses contribute equally
        self.lambda_phys = 1.0 / (ratio + 1e-8)
        self.lambda_data = 1.0

        return self.lambda_phys, self.lambda_data

    def get_weights(self):
        """Return current weights as dict."""
        return {"lambda_phys": self.lambda_phys, "lambda_data": self.lambda_data}


def create_loss_fn(loss_type: str = "mse", huber_delta: float = 0.1):
    """
    Create a loss function.

    Parameters
    ----------
    loss_type : str
        "mse", "mae", or "huber"
    huber_delta : float
        Delta parameter for Huber loss

    Returns
    -------
    callable
        Loss function (input, target) -> scalar loss
    """
    if loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "mae":
        return torch.nn.L1Loss()
    elif loss_type == "huber":
        return torch.nn.HuberLoss(delta=huber_delta, reduction="mean")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_per_point_residuals(
    outputs,
    targets,
    loss_fn_name: str = "mse",
):
    """
    Compute per-point residuals for adaptive sampling.

    Parameters
    ----------
    outputs : torch.Tensor
        Model outputs, shape (n, ...)
    targets : torch.Tensor
        Target values, shape (n, ...)
    loss_fn_name : str
        "mse", "mae", or "huber"

    Returns
    -------
    residuals : torch.Tensor
        Per-point residual magnitude, shape (n,)
    """
    if loss_fn_name == "mse":
        residuals = (outputs - targets) ** 2
    elif loss_fn_name == "mae":
        residuals = torch.abs(outputs - targets)
    elif loss_fn_name == "huber":
        delta = 0.1
        diff = torch.abs(outputs - targets)
        residuals = torch.where(
            diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_fn_name}")

    # Average across non-batch dimensions
    if residuals.dim() > 1:
        residuals = residuals.mean(dim=tuple(range(1, residuals.dim())))

    return residuals
