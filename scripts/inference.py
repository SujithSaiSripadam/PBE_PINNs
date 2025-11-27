import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src.models import PINN_CSD, PINN_CONC
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


def load_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def create_evaluation_grids(cfg, device):
    L_grid_phys = torch.linspace(0, cfg.physics.L_max,201, device=device) # cfg.physics.n_L_grid, device=device)
    t_eval = torch.linspace(0, cfg.physics.total_time, 100, device=device)  # dense time
    return L_grid_phys, t_eval


def evaluate_models(csd_net, conc_net, L_grid_phys, t_eval, cfg, device):
    csd_net.eval()
    conc_net.eval()
    with torch.no_grad():
        # Concentration
        t_norm = (t_eval / cfg.physics.t_scale).to(device)
        c_c_hat, c_wm_hat = conc_net(t_norm)
        c_c = (c_c_hat.cpu() * cfg.c_scale).numpy()
        c_wm = (c_wm_hat.cpu() * cfg.c_scale).numpy()

        # CSD
        T_mesh, L_mesh = torch.meshgrid(t_eval, L_grid_phys, indexing='ij')
        T_flat = T_mesh.reshape(-1)
        L_flat = L_mesh.reshape(-1)

        t_norm_flat = (T_flat / cfg.physics.t_scale).to(device)
        L_norm_flat = (L_flat / cfg.physics.L_scale).to(device)

        n_c_hat_flat, _ = csd_net(t_norm_flat, L_norm_flat)
        n_c_flat = (n_c_hat_flat.cpu() * cfg.physics.n_scale).numpy()

        n_c_2d = n_c_flat.reshape(T_mesh.shape)

        return {
            "t": t_eval.cpu().numpy(),
            "L": L_grid_phys.cpu().numpy(),
            "c_c": c_c,
            "c_wm": c_wm,
            "n_c": n_c_2d,
        }


def plot_concentration_vs_time(results, save_dir):
    plt.figure()
    plt.plot(results["t"], results["c_c"], label="Crystallizer", linewidth=2)
    plt.plot(results["t"], results["c_wm"], label="Wet Mill", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Concentration [kg/kg]")
    plt.title("Concentration vs Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "concentration_vs_time.png"))
    plt.close()

def plot_normalized_csd_vs_L(results, save_dir, time_points=None):
    """
    Plot normalized number density: n(L) / ∫n(L)dL  vs L [μm]
    """
    if time_points is None:
        time_points = [ 1200]  # seconds

    t = results["t"]
    L = results["L"]  # in meters
    L_um = L * 1e6    # convert to micrometers
    n_c = results["n_c"]  # shape: (n_t, n_L)

    dL = L[1] - L[0]  # uniform grid assumed

    plt.figure()
    for tp in time_points:
        idx = np.argmin(np.abs(t - tp))
        n_raw = n_c[idx, :]  # raw number density

        # Normalize: ∫ n(L) dL = 1
        integral = scipy.integrate.simpson(n_raw, L)  # or simpson, but trapz is fine
        if integral > 0:
            n_norm = n_raw / integral
        else:
            n_norm = n_raw

        plt.plot(L_um, n_norm, label=f"t = {int(t[idx])} s", linewidth=2)

    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Normalized Number Density $\\tilde{n}(L)$ [1/m]")
    plt.title("Normalized Crystal Size Distribution (CSD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_csd_vs_L.png"))
    plt.close()
    
def plot_csd_3d_surface(results, save_dir):
    """
    3D surface plot: time vs L vs n(L,t)
    """
    t = results["t"]               # [n_t]
    L = results["L"] * 1e6         # [n_L] → μm
    n_c = results["n_c"]           # [n_t, n_L]

    T, L_mesh = np.meshgrid(t, L, indexing='ij')  # T: [n_t, n_L], L_mesh: [n_t, n_L]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        T, L_mesh, n_c,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Crystal Size L [μm]")
    ax.set_zlabel("Number Density n(L,t) [#/m⁴]")
    ax.set_title("3D CSD Evolution")

    fig.colorbar(surf, shrink=0.5, aspect=15, label="n(L,t) [#/m⁴]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_3d_surface.png"), dpi=300)
    plt.close()

def plot_normalized_csd_3d_surface(results, save_dir):
    t = results["t"]
    L = results["L"]  # keep in meters for integration
    L_um = L * 1e6
    n_c = results["n_c"]  # [n_t, n_L]

    dL = L[1] - L[0]
    n_c_norm = np.zeros_like(n_c)

    for i in range(n_c.shape[0]):
        integral = np.trapz(n_c[i, :], L)
        if integral > 0:
            n_c_norm[i, :] = n_c[i, :] / integral
        else:
            n_c_norm[i, :] = 0.0

    T, L_mesh = np.meshgrid(t, L_um, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        T, L_mesh, n_c_norm,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Crystal Size L [μm]")
    ax.set_zlabel("Normalized Number Density $\\tilde{n}(L,t)$ [1/m]")
    ax.set_title("3D Normalized CSD Evolution")

    fig.colorbar(surf, shrink=0.5, aspect=15, label="$\\tilde{n}(L,t)$ [1/m]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_csd_3d_surface.png"), dpi=300)
    plt.close()
    
def plot_csd_snapshots(results, save_dir, time_points=None):
    if time_points is None:
        time_points = [1200]#[0, 300, 600, 900, 1200]  # seconds

    t = results["t"]
    L = results["L"] * 1e6  # convert to μm
    n_c = results["n_c"]
    
    print(f" n_c : {n_c} | ")

    plt.figure()
    for tp in time_points:
        idx = np.argmin(np.abs(t - tp))
        plt.plot(L, n_c[idx, :], label=f"t = {int(t[idx])} s", linewidth=1.8)

    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Number Density n(L) [#/m⁴]")
    plt.title("Crystal Size Distribution (CSD) at Selected Times")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_snapshots.png"))
    plt.close()


def plot_csd_heatmap(results, save_dir):
    t = results["t"]
    L = results["L"] * 1e6  # μm
    n_c = results["n_c"]

    plt.figure()
    pcm = plt.pcolormesh(t, L, n_c.T, shading='auto', cmap="viridis", norm=LogNorm())
    plt.colorbar(pcm, label="Number Density n(L,t) [#/m⁴]")
    plt.xlabel("Time [s]")
    plt.ylabel("Crystal Size L [μm]")
    plt.title("CSD Evolution (Heatmap)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "csd_heatmap.png"))
    plt.close()


def plot_mass_balance(results, cfg, save_dir):
    """
    Compute total crystal mass: M = rho_c * kv * ∫ L^3 n(L) dL
    """
    L = torch.tensor(results["L"], dtype=torch.float32)
    n_c = torch.tensor(results["n_c"], dtype=torch.float32)
    dL = L[1] - L[0]

    # Compute m3 = ∫ L^3 n(L) dL for each time
    L3 = L ** 3  # [n_L]
    m3 = torch.sum(L3.unsqueeze(0) * n_c, dim=1) * dL  # [n_t]

    crystal_mass = cfg.physics.rho_c * cfg.physics.kv * m3  # [kg/m³]

    plt.figure()
    plt.plot(results["t"], crystal_mass.numpy(), color="purple", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Crystal Mass Density [kg/m³]")
    plt.title("Crystal Mass Evolution")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "crystal_mass.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="/Users/sujithsaisripadam/PBE_PINNs/configs/configs.yaml", help="Path to config (for scales)")
    parser.add_argument("--output_dir", type=str, default="plots/", help="Directory to save plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config (minimal: just need scales and physics)
    import yaml
    print(args.config)
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)

    # Convert to simple object
    class Config:
        pass
    cfg = Config()
    cfg.physics = Config()
    int_keys = {"n_L_grid", "seed", "n_t_collocation"}
    float_keys = {
        "Vc", "Vwm", "kv", "rho_c", "kb", "b", "kg", "g", "kd", "d", "Ln",
        "a0", "a1", "a2", "rho_imp", "h_imp", "l_imp", "d_imp", "beta", "ka", "kf",
        "L_max", "total_time", "t_scale", "L_scale", "c_ref_temp", "n_scale"
    }

    for k, v in cfg_raw["physics"].items():
        if k in int_keys:
            setattr(cfg.physics, k, int(v))
        elif k in float_keys:
            setattr(cfg.physics, k, float(v))
        else:
            # Fallback: auto-detect
            if isinstance(v, (int, float)):
                setattr(cfg.physics, k, float(v))
            else:
                setattr(cfg.physics, k, v)

    # Now compute c_scale safely
    cfg.c_scale = float(
        cfg.physics.a0 + cfg.physics.a1 * cfg.physics.c_ref_temp + cfg.physics.a2 * (cfg.physics.c_ref_temp ** 2)
    )
    device = torch.device(args.device)

    # Load model
    ckpt = load_checkpoint(args.ckpt, device)
    L_grid = ckpt.get("L_grid", None)

    # Reconstruct models
    csd_net = PINN_CSD(hidden_dim=512, num_layers=5)  # must match training!
    conc_net = PINN_CONC(hidden_dim=512, num_layers=5)

    csd_net.load_state_dict(ckpt["csd_state_dict"])
    conc_net.load_state_dict(ckpt["conc_state_dict"])
    csd_net.to(device)
    conc_net.to(device)

    # Evaluation grids
    L_grid_phys, t_eval = create_evaluation_grids(cfg, device)

    # Forward evaluation
    results = evaluate_models(csd_net, conc_net, L_grid_phys, t_eval, cfg, device)

    # Generate plots
    plot_concentration_vs_time(results, args.output_dir)
    plot_csd_snapshots(results, args.output_dir)
    plot_csd_heatmap(results, args.output_dir)
    plot_mass_balance(results, cfg, args.output_dir)
    plot_normalized_csd_vs_L(results, args.output_dir)
    plot_csd_3d_surface(results, args.output_dir)
    plot_normalized_csd_3d_surface(results, args.output_dir)
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
