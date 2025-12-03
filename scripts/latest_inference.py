import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LogNorm
import scipy
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Plotting config
# ---------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "savefig.dpi": 300,
    "figure.figsize": (8, 6)
})

# ---------------------------
# Import project modules
# ---------------------------
try:
    from src.models import PINN_CSD, PINN_CONC
    from src.utils import load_data
    from src.physics.operators import supersaturation, nucleation_rate
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.models import PINN_CSD, PINN_CONC
    from src.utils import load_data
    from src.physics.operators import supersaturation, nucleation_rate


def load_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def create_evaluation_grids(cfg, device, nL=201):
    L_grid_phys = torch.linspace(0, cfg.physics.L_max, nL, device=device)
    t_eval = torch.linspace(0, cfg.physics.total_time, 100, device=device)
    return L_grid_phys, t_eval


def plot_normalized_csd_vs_L(L, n_c, save_dir, time_label="Final"):
    L_um = L * 1e6
    integral = scipy.integrate.simpson(n_c, L)
    n_norm = n_c / integral if integral > 0 else n_c

    plt.figure()
    plt.plot(L_um, n_norm, linewidth=2)
    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Normalized Number Density $\\tilde{n}(L)$ [1/m]")
    plt.title(f"Normalized CSD at {time_label} Time (Plot 12)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_csd_vs_L.png"))
    plt.close()
    return n_norm, L


def plot_csd_snapshots(L, n_c, save_dir, time_label="Final"):
    L_um = L * 1e6
    plt.figure()
    plt.plot(L_um, n_c, linewidth=1.8)
    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Number Density n(L) [#/m⁴]")
    plt.title(f"CSD at {time_label} Time")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_snapshots.png"))
    plt.close()

def plot_target_vs_product(L, n_pred, save_dir, target_params):
    L_um = L * 1e6
    mu = np.log(target_params["median_L"])
    sigma = target_params["spread"]
    n_target = (1 / (L * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(L) - mu) / sigma) ** 2)
    n_target = np.where(L > 0, n_target, 0)
    n_target /= np.trapz(n_target, L)

    plt.figure(figsize=(8, 5))
    plt.plot(L_um, n_target, 'k--', label="Target", linewidth=2)
    plt.plot(L_um, n_pred, 'b-', label="Product", linewidth=2)
    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Normalized Number Density")
    plt.title("Figure 5: Target vs Product CSD")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure5_target_vs_product.png"))
    plt.close()

def plot_csd_heatmap(t, L, n_c_2d, save_dir):
    L_um = L * 1e6
    plt.figure()
    pcm = plt.pcolormesh(t, L_um, n_c_2d.T, shading='auto', cmap="viridis", norm=LogNorm())
    plt.colorbar(pcm, label="Number Density n(L,t) [#/m⁴]")
    plt.xlabel("Time [s]")
    plt.ylabel("Crystal Size L [μm]")
    plt.title("CSD Evolution (Heatmap)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_heatmap.png"))
    plt.close()

def plot_csd_3d_surface(t, L, n_c_2d, save_dir):
    L_um = L * 1e6
    T, L_mesh = np.meshgrid(t, L_um, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, L_mesh, n_c_2d, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Crystal Size L [μm]")
    ax.set_zlabel("Number Density n(L,t) [#/m⁴]")
    ax.set_title("3D CSD Evolution")
    fig.colorbar(surf, shrink=0.5, aspect=15, label="n(L,t) [#/m⁴]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_3d_surface.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/configs.yaml", help="Path to config")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to data.csv")
    parser.add_argument("--output_dir", type=str, default="plots/inference", help="Directory to save plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)

    class Config: pass
    cfg = Config()
    cfg.physics = Config()

    int_keys = {"n_L_grid", "seed", "n_t_collocation"}
    float_keys = {
        "Vc", "Vwm", "kv", "rho_c", "kb", "b", "kg", "g", "kd", "d", "Ln",
        "a0", "a1", "a2", "rho_imp", "h_imp", "l_imp", "d_imp", "beta", "ka", "kf",
        "L_max", "total_time", "t_scale", "L_scale", "c_ref_temp", "n_scale",
        "T_scale", "F_scale", "N_scale"
    }

    for k, v in cfg_raw["physics"].items():
        if k in int_keys:
            setattr(cfg.physics, k, int(v))
        elif k in float_keys:
            setattr(cfg.physics, k, float(v))
        else:
            if isinstance(v, (int, float)):
                setattr(cfg.physics, k, float(v))
            else:
                setattr(cfg.physics, k, v)

    cfg.c_scale = float(
        cfg.physics.a0 + cfg.physics.a1 * cfg.physics.c_ref_temp + cfg.physics.a2 * (cfg.physics.c_ref_temp ** 2)
    )
    device = torch.device(args.device)

    # Load model
    ckpt = load_checkpoint(args.ckpt, device)
    csd_net = PINN_CSD(hidden_dim=512)
    conc_net = PINN_CONC(hidden_dim=512)
    csd_net.load_state_dict(ckpt["csd_state_dict"])
    conc_net.load_state_dict(ckpt["conc_state_dict"])
    csd_net.to(device)
    conc_net.to(device)

    # Load data
    data = load_data(args.data_csv, nrows=None, device=device)
    t_true = data["t"]
    c_true = data["c"]
    cwm_true = data["cwm"]
    T_true = data["T"]
    F_true = data["F"]
    N_true = data["N"]

    # Predict concentration
    with torch.no_grad():
        t_norm = (t_true / cfg.physics.t_scale).to(device)
        T_norm = (T_true / cfg.physics.T_scale).to(device)
        F_norm = (F_true / cfg.physics.F_scale).to(device)
        N_norm = (N_true / cfg.physics.N_scale).to(device)
        c_c_hat, c_wm_hat = conc_net(t_norm, T_norm, F_norm, N_norm)
        c_pred = (c_c_hat * cfg.c_scale).cpu()
        cwm_pred = (c_wm_hat * cfg.c_scale).cpu()

    # Save predictions
    with open(os.path.join(args.output_dir, "PredsvsTrue.txt"), "w") as file:
        for i in range(len(t_true)):
            file.write(f"t={t_true[i].item():.1f}s | c_pred={c_pred[i].item():.6f} | c_true={c_true[i].item():.6f}\n")

    np.savetxt(
        os.path.join(args.output_dir, "concentration_pred_true.csv"),
        np.column_stack((t_true.cpu().numpy(), c_pred.numpy(), c_true.cpu().numpy(), cwm_pred.numpy(), cwm_true.cpu().numpy())),
        delimiter=",",
        header="t,c_pred,c_true,cwm_pred,cwm_true",
        comments=""
    )

    # Plot concentration (Plot 6 & 7)
    for (name, pred, true, title, fname) in [
        ("Crystallizer", c_pred, c_true, "Crystallizer: Predicted vs True", "c_crystallizer_pred_true.png"),
        ("Wet Mill", cwm_pred, cwm_true, "Wet Mill: Predicted vs True", "c_wetmill_pred_true.png")
    ]:
        plt.figure()
        plt.plot(t_true.cpu(), pred, label=f"{name} (Pred)", linewidth=2)
        plt.plot(t_true.cpu(), true.cpu(), '--', label=f"{name} (True)", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Concentration [kg/kg]")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname))
        plt.close()

    # Evaluate full CSD evolution
    L_grid_phys, t_eval = create_evaluation_grids(cfg, device, nL=501)
    nL = L_grid_phys.shape[0]
    n_t = t_eval.shape[0]

    # Interpolate T, F, N to t_eval using original data
    from scipy.interpolate import interp1d
    t_data = data["t"].cpu().numpy()
    T_data = data["T"].cpu().numpy()
    F_data = data["F"].cpu().numpy()
    N_data = data["N"].cpu().numpy()

    T_interp = interp1d(t_data, T_data, kind='linear', fill_value="extrapolate")
    F_interp = interp1d(t_data, F_data, kind='linear', fill_value="extrapolate")
    N_interp = interp1d(t_data, N_data, kind='linear', fill_value="extrapolate")

    n_c_2d = np.zeros((n_t, nL))
    n_wm_2d = np.zeros((n_t, nL))
    c_c_eval = np.zeros((n_t,))
    B_nuc_eval = np.zeros((n_t,))
    for i, t_val in enumerate(t_eval.cpu().numpy()):
        T_val = T_interp(t_val).item()
        F_val = F_interp(t_val).item()
        N_val = N_interp(t_val).item()

        t_vec = torch.full((nL,), t_val, device=device, dtype=torch.float32)
        T_vec = torch.full((nL,), T_val, device=device, dtype=torch.float32)
        F_vec = torch.full((nL,), F_val, device=device, dtype=torch.float32)
        N_vec = torch.full((nL,), N_val, device=device, dtype=torch.float32)

        t_norm_vec = t_vec / cfg.physics.t_scale
        L_norm_vec = L_grid_phys / cfg.physics.L_scale
        T_norm_vec = T_vec / cfg.physics.T_scale
        F_norm_vec = F_vec / cfg.physics.F_scale
        N_norm_vec = N_vec / cfg.physics.N_scale

        with torch.no_grad():
            n_hat, n_wm_hat = csd_net(t_norm_vec, L_norm_vec, T_norm_vec, F_norm_vec, N_norm_vec)
            n_c_2d[i, :] = (n_hat * cfg.physics.n_scale).cpu().numpy()
            n_wm_2d[i, :] = (n_wm_hat * cfg.physics.n_scale).cpu().numpy()

            # evaluate concentration (use single-point conc_net) for nucleation diagnostics
            t_single = torch.tensor([t_val], device=device, dtype=torch.float32)
            T_single = torch.tensor([T_val], device=device, dtype=torch.float32)
            F_single = torch.tensor([F_val], device=device, dtype=torch.float32)
            N_single = torch.tensor([N_val], device=device, dtype=torch.float32)
            t_norm_single = t_single / cfg.physics.t_scale
            T_norm_single = T_single / cfg.physics.T_scale
            F_norm_single = F_single / cfg.physics.F_scale
            N_norm_single = N_single / cfg.physics.N_scale
            c_hat_single, _ = conc_net(t_norm_single, T_norm_single, F_norm_single, N_norm_single)
            c_phys_single = (c_hat_single * cfg.c_scale).cpu().numpy()[0]
            c_c_eval[i] = c_phys_single

            # compute nucleation rate at the nucleus size (scalar)
            # use torch for operators
            c_t = torch.tensor(c_phys_single, device=device, dtype=torch.float32)
            T_t = torch.tensor(T_val, device=device, dtype=torch.float32)
            sigma_t = supersaturation(c_t, T_t, cfg.physics.a0, cfg.physics.a1, cfg.physics.a2)
            B_t = nucleation_rate(sigma_t, c_t, T_t, cfg.physics.kb, cfg.physics.b, cfg.physics.a0, cfg.physics.a1, cfg.physics.a2)
            B_nuc_eval[i] = float(B_t.cpu().numpy())

    t_np = t_eval.cpu().numpy()
    L_np = L_grid_phys.cpu().numpy()

    # --- Compute total number of crystals for each time and save to CSV ---
    # Bin width (h_l) on physical L grid
    dL = float((L_grid_phys[1] - L_grid_phys[0]).cpu().numpy())
    # N_total = Vc * sum_l n_c(L_l,t) * h_l
    N_total_crys = float(cfg.physics.Vc) * np.sum(n_c_2d * dL, axis=1)
    N_total_wm = float(cfg.physics.Vwm) * np.sum(n_wm_2d * dL, axis=1)
    

    # Save total crystals to CSV: columns = t, N_total
    np.savetxt(
        os.path.join(args.output_dir, "total_crystals.csv"),
        np.column_stack((t_np,n_c_2d, n_wm_2d,N_total_crys,N_total_wm)),
        delimiter=",",
        header="t,n(Crys),n(WM)N_crys,N_wm",
        comments=""
    )

    # --- Plot total number of crystals vs time ---
    plt.figure()
    plt.plot(t_np, N_total_crys, color="tab:purple", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Total Number of Crystals")
    plt.title("Total Number of Crystals in Crystallizer vs Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "total_crystals_crys-vs_time.png"))
    plt.close()
    
    plt.figure()
    plt.plot(t_np, N_total_wm, color="tab:purple", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Total Number of Crystals")
    plt.title("Total Number of Crystals in Wet Mill vs Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "total_crystals_wm_vs_time.png"))
    plt.close()

    # Plot 12: Normalized CSD at final time
    n_final = n_c_2d[-1, :]
    n_norm, L_used = plot_normalized_csd_vs_L(L_np, n_final, args.output_dir, time_label="Final")

    # Plot raw CSD (log)
    plot_csd_snapshots(L_np, n_final, args.output_dir, time_label="Final")

    # Figure 5: Target vs Product
    plot_target_vs_product(L_np, n_norm, args.output_dir, {"median_L": 300e-6, "spread": 0.8})

    # Additional visualizations
    plot_csd_heatmap(t_np, L_np, n_c_2d, args.output_dir)
    plot_csd_3d_surface(t_np, L_np, n_c_2d, args.output_dir)

    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()