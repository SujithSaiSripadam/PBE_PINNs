import torch
import math
from src.physics.operators import (
    solubility,
    supersaturation,
    nucleation_rate,
    growth_rate,
    selection_function,
)
from src.utils.grids import (
    gaussian_delta_on_grid,
    build_B_matrix,
    simpson_integrate_over_L,
)


class PhysicsLossFullPBM_Nondim:
    def __init__(self, L_grid_phys, cfg, c_scale, device="cpu", dtype=torch.float32):
        """
        Physics-informed loss for dimensionless two-compartment PBM.

        Parameters
        ----------
        L_grid_phys : torch.Tensor
            Physical length grid [m], shape (nL,)
        cfg : DictConfig or dict
            Hydra config containing all physics parameters under cfg.physics
        c_scale : float
            Reference concentration scale [kg/kg]
        device : str or torch.device
        dtype : torch.dtype
        """
        self.device = torch.device(device)
        self.dtype = dtype

        self.L_grid_phys = L_grid_phys.to(self.device).to(self.dtype)
        self.nL = self.L_grid_phys.numel()
        self.dL_phys = (self.L_grid_phys[1] - self.L_grid_phys[0]).item()

        # Store config parameters for clarity
        p = cfg
        self.Vc = p.Vc
        self.Vwm = p.Vwm
        self.kv = p.kv
        self.rho_c = p.rho_c
        self.kb = p.kb
        self.b = p.b
        self.kg = p.kg
        self.g = p.g
        self.kd = p.kd
        self.d = p.d
        self.Ln = p.Ln
        self.a0 = p.a0
        self.a1 = p.a1
        self.a2 = p.a2
        self.rho_imp = p.rho_imp
        self.h_imp = p.h_imp
        self.l_imp = p.l_imp
        self.d_imp = p.d_imp
        self.beta = p.beta
        self.ka = p.ka
        self.kf = p.kf

        # Normalization scales
        self.t_scale = p.t_scale
        self.L_scale = p.L_scale
        self.n_scale = p.n_scale
        self.c_scale = c_scale

        # Precompute physical delta and B matrix
        self.delta_phys = gaussian_delta_on_grid(
            self.L_grid_phys, self.Ln, self.dL_phys
        ).to(self.device).to(self.dtype)

        self.B_phys = build_B_matrix(
            self.L_grid_phys, self.dL_phys, self.Ln, self.ka, self.kf
        ).to(self.device).to(self.dtype)

        # Convert to dimensionless kernels
        self.delta_hat = (self.delta_phys * self.L_scale).to(self.dtype)
        self.B_hat = (self.B_phys * self.L_scale).to(self.dtype)

    def compute_loss(self, csd_net, conc_net, t_coll_phys, L_coll_phys, T_coll_phys, F_coll_phys, N_coll_phys):
        """
        Compute total physics loss for a batch of collocation points.

        All inputs are in physical units.
        Networks accept normalized inputs and output dimensionless fields.
        """
        t_coll_phys = t_coll_phys.to(self.device).to(self.dtype)
        L_coll_phys = L_coll_phys.to(self.device).to(self.dtype)
        T_coll_phys = T_coll_phys.to(self.device).to(self.dtype)
        F_coll_phys = F_coll_phys.to(self.device).to(self.dtype)
        N_coll_phys = N_coll_phys.to(self.device).to(self.dtype)

        # Normalize inputs
        t_hat = (t_coll_phys / self.t_scale).requires_grad_(True)
        L_hat = (L_coll_phys / self.L_scale).requires_grad_(True)

        # Forward pass
        n_c_hat, n_wm_hat = csd_net(t_hat, L_hat)
        c_c_hat, c_wm_hat = conc_net(t_hat)

        # Reconstruct physical concentrations
        c_c_phys = c_c_hat * self.c_scale
        c_wm_phys = c_wm_hat * self.c_scale

        # Compute supersaturation and physics terms
        sigma_c = supersaturation(c_c_phys, T_coll_phys, self.a0, self.a1, self.a2)
        B_nuc_dim = nucleation_rate(
            sigma_c, c_c_phys, T_coll_phys, self.kb, self.b, self.a0, self.a1, self.a2
        )
        G_dim = growth_rate(
            sigma_c, c_c_phys, T_coll_phys, self.kg, self.g, self.kd, self.d, self.a0, self.a1, self.a2
        )
        S_dim = selection_function(
            L_coll_phys, N_coll_phys, self.rho_imp, self.h_imp, self.l_imp, self.d_imp, self.beta
        )

        # Guard against NaNs/Infs
        B_nuc_dim = torch.nan_to_num(B_nuc_dim, nan=0.0, posinf=0.0, neginf=0.0)
        G_dim = torch.nan_to_num(G_dim, nan=0.0, posinf=0.0, neginf=0.0)
        S_dim = torch.nan_to_num(S_dim, nan=0.0, posinf=0.0, neginf=0.0)

        # Dimensionless coefficients
        G_hat = G_dim * (self.t_scale / self.L_scale)
        B_nuc_hat = B_nuc_dim * (self.t_scale / self.n_scale)
        S_hat = S_dim * self.t_scale
        Fhat_cryst = (F_coll_phys / self.Vc) * self.t_scale
        Fhat_wm = (F_coll_phys / self.Vwm) * self.t_scale

        # Map collocation L to delta on grid
        with torch.no_grad():
            diff = torch.abs(L_coll_phys.view(-1, 1) - self.L_grid_phys.view(1, -1))
            nearest_idx = torch.argmin(diff, dim=1)
        delta_hat_at_coll = self.delta_hat[nearest_idx]

        # --- Crystallizer PDE ---
        ones = torch.ones_like(n_c_hat)
        dn_hat_dtau = torch.autograd.grad(n_c_hat, t_hat, grad_outputs=ones, create_graph=True)[0]
        Gn_hat = G_hat * n_c_hat
        dGnhat_dlambda = torch.autograd.grad(Gn_hat, L_hat, grad_outputs=ones, create_graph=True)[0]
        
        # Clean up intermediate (don't keep in graph)
        #del Gn_hat

        pde_cryst_hat = (
            dn_hat_dtau
            + dGnhat_dlambda
            - B_nuc_hat * delta_hat_at_coll
            - Fhat_cryst * (n_wm_hat - n_c_hat)
        )
        pde_cryst_loss = torch.mean(pde_cryst_hat ** 2)
        
        # Delete gradient tensors to free memory
        #del dn_hat_dtau, dGnhat_dlambda, pde_cryst_hat, ones

        # --- Crystallizer Mass Balance ---
        t_vals_phys = t_coll_phys.detach()
        uniq_t, inv_idx = torch.unique(t_vals_phys, return_inverse=True)
        n_unique = uniq_t.shape[0]

        m2_phys = torch.zeros(n_unique, device=self.device, dtype=self.dtype)
        m3_phys = torch.zeros(n_unique, device=self.device, dtype=self.dtype)

        for k in range(n_unique):
            mask = inv_idx == k
            if mask.any():
                n_L_vec_hat = torch.zeros(self.nL, device=self.device, dtype=self.dtype)
                idxs = nearest_idx[mask]
                vals_hat = n_c_hat[mask]
                n_L_vec_hat.index_add_(0, idxs, vals_hat)
                n_L_vec_phys = n_L_vec_hat * self.n_scale
                m2_phys[k] = simpson_integrate_over_L(
                    (self.L_grid_phys ** 2).unsqueeze(0) * n_L_vec_phys.unsqueeze(0), self.dL_phys
                )
                m3_phys[k] = simpson_integrate_over_L(
                    (self.L_grid_phys ** 3).unsqueeze(0) * n_L_vec_phys.unsqueeze(0), self.dL_phys
                )

        m2_full = m2_phys[inv_idx]
        m3_full = m3_phys[inv_idx]
        theta_c = self.kv * 1e-18 * m3_full

        dc_hat_dtau = torch.autograd.grad(
            c_c_hat, t_hat, grad_outputs=torch.ones_like(c_c_hat), create_graph=True
        )[0]

        RHS_dim = (self.rho_c * self.kv / (1.0 - theta_c + 1e-12)) * (
            3.0 * G_dim * m2_full + B_nuc_dim * (self.Ln ** 3)
        )
        RHS_dim = RHS_dim - (F_coll_phys / self.Vc) * (c_wm_phys - c_c_phys)
        mass_cryst_hat = dc_hat_dtau + (self.t_scale / self.c_scale) * RHS_dim
        mass_cryst_loss = torch.mean(mass_cryst_hat ** 2)

        # --- Wet Mill PDE ---
        t_rep = uniq_t.view(-1, 1).repeat(1, self.nL).view(-1)
        L_rep = self.L_grid_phys.repeat(n_unique)
        t_rep_norm = t_rep / self.t_scale
        L_rep_norm = L_rep / self.L_scale
        _, n_wm_rep_hat = csd_net(t_rep_norm, L_rep_norm)
        n_wm_matrix_phys = (n_wm_rep_hat * self.n_scale).view(n_unique, self.nL)
        
        # Clean up intermediate tensors to reduce memory
        #del t_rep, L_rep, t_rep_norm, L_rep_norm, n_wm_rep_hat

        N_per_t = torch.zeros(n_unique, device=self.device, dtype=self.dtype)
        for k in range(n_unique):
            mask = inv_idx == k
            if mask.any():
                N_per_t[k] = N_coll_phys[mask].mean()
        N_mat = N_per_t.view(-1, 1).repeat(1, self.nL)
        S_mat_dim = selection_function(
            self.L_grid_phys.view(1, -1), N_mat, self.rho_imp, self.h_imp,
            self.l_imp, self.d_imp, self.beta
        )

        integrals_dim = torch.zeros((n_unique, self.nL), device=self.device, dtype=self.dtype)
        for k in range(n_unique):
            w_parent = n_wm_matrix_phys[k, :] * S_mat_dim[k, :]
            integrals_dim[k, :] = torch.matmul(w_parent, self.B_phys) * self.dL_phys
            #del w_parent  # Clean up loop intermediate

        integrals_hat = integrals_dim * (self.t_scale / self.n_scale)
        integrals_coll_hat = integrals_hat[inv_idx, nearest_idx]
        S_coll_hat = S_mat_dim[inv_idx, nearest_idx] * self.t_scale
        recirc_wm_hat = Fhat_wm * (n_c_hat - n_wm_hat)
        
        # Clean up large matrices
        #del integrals_dim, S_mat_dim

        dn_wm_hat_dtau = torch.autograd.grad(
            n_wm_hat, t_hat, grad_outputs=torch.ones_like(n_wm_hat), create_graph=True
        )[0]
        pde_wm_hat = dn_wm_hat_dtau - (integrals_coll_hat - S_coll_hat * n_wm_hat + recirc_wm_hat)
        pde_wm_loss = torch.mean(pde_wm_hat ** 2)
        
        # Clean up
        #del dn_wm_hat_dtau, pde_wm_hat, integrals_coll_hat, S_coll_hat, recirc_wm_hat

        # --- Wet Mill Mass Balance ---
        dc_wm_hat_dtau = torch.autograd.grad(
            c_wm_hat, t_hat, grad_outputs=torch.ones_like(c_wm_hat), create_graph=True, retain_graph=False
        )[0]
        RHS_dim_wm = - (F_coll_phys / self.Vwm) * (c_c_phys - c_wm_phys)
        mass_wm_hat = dc_wm_hat_dtau - (self.t_scale / self.c_scale) * RHS_dim_wm
        mass_wm_loss = torch.mean(mass_wm_hat ** 2)
        
        # Clean up
        #del dc_wm_hat_dtau, mass_wm_hat, RHS_dim_wm

        # --- Boundary & Initial Conditions ---
        mask_Lmax = nearest_idx == (self.nL - 1)
        bc_loss = torch.mean((n_c_hat[mask_Lmax] ** 2)) + torch.mean((n_wm_hat[mask_Lmax] ** 2)) if mask_Lmax.any() else torch.tensor(0.0, device=self.device)

        mask_t0 = t_coll_phys == 0.0
        if mask_t0.any():
            ic_loss = torch.mean(n_c_hat[mask_t0] ** 2) + torch.mean(n_wm_hat[mask_t0] ** 2)
            ic_mass_loss = torch.mean((c_c_hat[mask_t0] - c_wm_hat[mask_t0]) ** 2)
        else:
            ic_loss = torch.tensor(0.0, device=self.device)
            ic_mass_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = (
            pde_cryst_loss
            + pde_wm_loss
            + mass_cryst_loss
            + mass_wm_loss
            + bc_loss
            + ic_loss
            + ic_mass_loss
        )

        # Predictions (physical units)
        preds = {
            "conc_c": (c_c_hat * self.c_scale).detach(),
            "numden_c": (n_c_hat * self.n_scale).detach(),
            "conc_wm": (c_wm_hat * self.c_scale).detach(),
            "numden_wm": (n_wm_hat * self.n_scale).detach(),
        }

        # Metrics
        metrics = {
            "pde_cryst_loss": pde_cryst_loss.item(),
            "pde_wm_loss": pde_wm_loss.item(),
            "mass_cryst_loss": mass_cryst_loss.item(),
            "mass_wm_loss": mass_wm_loss.item(),
            "bc_nLmax_loss": bc_loss.item(),
        }

        return loss, metrics, preds
