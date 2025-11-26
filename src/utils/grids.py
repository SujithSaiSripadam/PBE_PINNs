import torch
import math
# ---------------------------
# Delta approximation on physical grid (normalized later)
# ---------------------------
def gaussian_delta_on_grid(L_grid, Ln, dL, eps=None):
    if eps is None:
        eps = 0.5 * (dL if isinstance(dL, float) else dL.item())
    eps_t = torch.as_tensor(eps, device=L_grid.device, dtype=L_grid.dtype)
    unscaled = torch.exp(-0.5 * ((L_grid - Ln) / eps_t) ** 2) / (eps_t * math.sqrt(2.0 * math.pi))
    s = unscaled.sum()
    if s == 0:
        idx = torch.argmin(torch.abs(L_grid - Ln))
        delta = torch.zeros_like(L_grid)
        delta[idx] = 1.0 / (dL if isinstance(dL, float) else dL.item())
        return delta
    delta = unscaled / (s * (dL if isinstance(dL, float) else dL.item()))
    return delta

# ---------------------------
# Build B matrix — returns B_dim(L_parent_index, L_daughter_index)
# --> Convert to dimensionless inside loss.
# ---------------------------
def build_B_matrix(L_grid, dL, Ln, ka_local, kf_local):
    nL = L_grid.numel()
    L_row = L_grid.view(1, nL)
    Lambda_col = L_grid.view(nL, 1)
    sigma_f = (Lambda_col / 100.0).clamp(min=1e-12)
    exponent = -0.5 * ((L_row - (Lambda_col / 2.0)) / sigma_f) ** 2
    frag_pdf = torch.exp(exponent) / (sigma_f * math.sqrt(2.0 * math.pi))
    b_f = kf_local * frag_pdf
    delta_vec = gaussian_delta_on_grid(L_grid, Ln, dL)
    b_a = ka_local * Lambda_col * delta_vec.view(1, nL)
    B_raw = b_a + b_f
    B_raw = torch.clamp(B_raw, 0.0, 1e12)
    # mass normalize per parent (discrete)
    L3 = (L_grid ** 3).view(1, nL)
    numerator = (L3 * B_raw).sum(dim=1) * dL
    target = (L_grid ** 3)
    numerator = numerator.clamp(min=1e-30)
    scale = torch.where(numerator > 0, target / numerator, torch.ones_like(numerator))
    B = B_raw * scale.view(-1, 1)
    B = torch.clamp(B, 0.0, 1e18)
    return B

# ---------------------------
# Simpson integration helper (on physical L grid)
# ---------------------------
def simpson_integrate_over_L(f_vals, dL):
    if f_vals.dim() == 1:
        f = f_vals.unsqueeze(0)
        squeeze = True
    else:
        f = f_vals
        squeeze = False
    nL = f.shape[1]
    if nL % 2 == 0:
        raise ValueError("Simpson requires odd nL")
    w = torch.ones(nL, device=f.device, dtype=f.dtype)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    integral = (dL / 3.0) * (f * w).sum(dim=1)
    if squeeze:
        return integral[0]
    return integral