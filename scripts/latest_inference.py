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
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.models import PINN_CSD, PINN_CONC
    from src.utils import load_data


def load_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def create_evaluation_grids(cfg, device, nL=201):
    L_grid_phys = torch.linspace(0, cfg.physics.L_max, nL, device=device)
    t_eval = torch.linspace(0, cfg.physics.total_time, 100, device=device)
    return L_grid_phys, t_eval


def evaluate_models(csd_net, conc_net, L_grid_phys, t_eval, cfg, device):
    csd_net.eval()
    conc_net.eval()
    with torch.no_grad():
        t_norm = (t_eval / cfg.physics.t_scale).to(device)
        c_c_hat, c_wm_hat = conc_net(t_norm)
        c_c = (c_c_hat.cpu() * cfg.c_scale).numpy()
        c_wm = (c_wm_hat.cpu() * cfg.c_scale).numpy()

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


def plot_normalized_csd_vs_L(results, save_dir, time_points=None):
    if time_points is None:
        time_points = [1200.0]

    t = results["t"]
    L = results["L"]
    L_um = L * 1e6
    n_c = results["n_c"]

    plt.figure()
    for tp in time_points:
        idx = np.argmin(np.abs(t - tp))
        n_raw = n_c[idx, :]
        integral = scipy.integrate.simpson(n_raw, L)
        n_norm = n_raw / integral if integral > 0 else n_raw
        plt.plot(L_um, n_norm, label=f"t = {int(t[idx])} s", linewidth=2)

    plt.xlabel("Crystal Size L [μm]")
    plt.ylabel("Normalized Number Density $\\tilde{n}(L)$ [1/m]")
    plt.title("Normalized Crystal Size Distribution (CSD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_csd_vs_L.png"))
    plt.close()


def plot_csd_snapshots(results, save_dir, time_points=None):
    if time_points is None:
        time_points = [1200.0]

    t = results["t"]
    L = results["L"] * 1e6
    n_c = results["n_c"]

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
        "L_max", "total_time", "t_scale", "L_scale", "c_ref_temp", "n_scale"
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

    # Load true data
    data = load_data(args.data_csv, nrows=None, device=device)
    t_true = data["t"]
    c_true = data["c"]
    cwm_true = data["cwm"]

    # Predict at true timestamps
    with torch.no_grad():
        t_norm = (t_true / cfg.physics.t_scale).to(device)
        c_c_hat, c_wm_hat = conc_net(t_norm)
        c_pred = (c_c_hat * cfg.c_scale).cpu()
        cwm_pred = (c_wm_hat * cfg.c_scale).cpu()

    # Print predictions vs truth
    print("\n=== Concentration: Predicted vs True (first 20 points) ===")
    with open("PredsvsTrue.txt", "w") as file:
        for i in range(len(t_true)):
            file.write(f"t={t_true[i].item():.1f}s | c_pred={c_pred[i].item():.6f} | c_true={c_true[i].item():.6f}\n")
            
        

    # Save full comparison
    comparison = np.column_stack((
        t_true.cpu().numpy(),
        c_pred.numpy(),
        c_true.cpu().numpy(),
        cwm_pred.numpy(),
        cwm_true.cpu().numpy()
    ))
    np.savetxt(
        os.path.join(args.output_dir, "concentration_pred_true.csv"),
        comparison,
        delimiter=",",
        header="t,c_pred,c_true,cwm_pred,cwm_true",
        comments=""
    )

    # Plot like Plot 6 & 7
    plt.figure()
    plt.plot(t_true.cpu(), c_pred, label="Crystallizer (Pred)", linewidth=2)
    plt.plot(t_true.cpu(), c_true.cpu(), '--', label="Crystallizer (True)", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Concentration [kg/kg]")
    plt.title("Crystallizer: Predicted vs True Concentration")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "c_crystallizer_pred_true.png"))
    plt.close()

    plt.figure()
    plt.plot(t_true.cpu(), cwm_pred, label="Wet Mill (Pred)", linewidth=2)
    plt.plot(t_true.cpu(), cwm_true.cpu(), '--', label="Wet Mill (True)", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Concentration [kg/kg]")
    plt.title("Wet Mill: Predicted vs True Concentration")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "c_wetmill_pred_true.png"))
    plt.close()

    # Plot CSD at final time (like Plot 12)
    L_grid_phys, _ = create_evaluation_grids(cfg, device)
    t_final = torch.tensor([float(t_true[-1].item())], device=device)
    results_fine = evaluate_models(csd_net, conc_net, L_grid_phys, t_final, cfg, device)

    plot_normalized_csd_vs_L(results_fine, args.output_dir, time_points=[t_final.item()])
    plot_csd_snapshots(results_fine, args.output_dir, time_points=[t_final.item()])

    print(f"All plots and comparison saved to: {args.output_dir}")


if __name__ == "__main__":
    main()