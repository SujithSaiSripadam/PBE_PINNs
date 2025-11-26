import torch
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from src.models import PINN_CSD, PINN_CONC
from src.physics import PhysicsLossFullPBM_Nondim
from src.utils import load_data
from src.utils.grids import simpson_integrate_over_L
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
import gc

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def cleanup_memory():
    """Force garbage collection and CUDA cache clear"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@hydra.main(config_path="../configs", config_name="configs", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed)

    # Compute c_scale from config
    c_scale = (cfg.physics.a0 + cfg.physics.a1 * cfg.physics.c_ref_temp + 
               cfg.physics.a2 * cfg.physics.c_ref_temp**2)

    # Build L grid
    L_grid_phys = torch.linspace(0.0, cfg.physics.L_max, cfg.physics.n_L_grid, 
                                 device=device, dtype=torch.float32)

    # Load data
    data = load_data(cfg.data.csv_path, cfg.data.nrows, device=device)

    # Subsample time for collocation
    n_t_colloc = min(cfg.physics.n_t_collocation, len(data["t"]))
    idx = torch.linspace(0, len(data["t"])-1, n_t_colloc).long()
    t_small = data["t"][idx]
    T_small = data["T"][idx]
    F_small = data["F"][idx]
    N_small = data["N"][idx]

    # Create collocation mesh
    T_grid_mesh, L_grid_mesh = torch.meshgrid(t_small, L_grid_phys, indexing='ij')
    t_coll = T_grid_mesh.reshape(-1)
    L_coll = L_grid_mesh.reshape(-1)
    T_all = T_small.repeat_interleave(cfg.physics.n_L_grid)
    F_all = F_small.repeat_interleave(cfg.physics.n_L_grid)
    N_all = N_small.repeat_interleave(cfg.physics.n_L_grid)

    colloc_dataset = TensorDataset(t_coll, L_coll, T_all, F_all, N_all)
    colloc_loader = DataLoader(
        colloc_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Models
    csd_net = PINN_CSD(
        hidden_dim=cfg.model.hidden_dim,
        activation=cfg.model.activation,
        num_layers=cfg.model.num_layers
    ).to(device)
    conc_net = PINN_CONC(
        hidden_dim=cfg.model.hidden_dim,
        activation=cfg.model.activation,
        num_layers=cfg.model.num_layers
    ).to(device)

    # Physics loss (pass full cfg.physics dict)
    phys = PhysicsLossFullPBM_Nondim(
        L_grid_phys=L_grid_phys,
        cfg=cfg.physics,
        c_scale=c_scale,
        device=device,
        dtype=torch.float32
    )

    # Optimizer
    params = list(csd_net.parameters()) + list(conc_net.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # Logging
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name= hydra.core.hydra_config.HydraConfig.get().job.name,
            settings=wandb.Settings(verbosity=wandb.Verbosity.ERROR)
        )
        wandb.watch((csd_net, conc_net), log="all", log_freq=cfg.logging.log_freq)

    # ---------------------------
    # Training Loop
    # ---------------------------
    global_step = 0
    writer = SummaryWriter(log_dir="runs/" + hydra.core.hydra_config.HydraConfig.get().job.name)

    for epoch in range(cfg.training.n_epochs):
        csd_net.train()
        conc_net.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(colloc_loader, desc=f"Epoch {epoch+1}/{cfg.training.n_epochs}", leave=False)
        for batch in pbar:
            t_b, L_b, T_b, F_b, N_b = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss_phys, loss_dict, preds = phys.compute_loss(csd_net, conc_net, t_b, L_b, T_b, F_b, N_b)
            loss_value = loss_phys.item()
            loss_phys.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip)

            optimizer.step()

            # Accumulate
            epoch_loss += loss_value
            num_batches += 1
            global_step += 1
            
            # Clean up batch and loss to free memory
            del t_b, L_b, T_b, F_b, N_b, loss_phys, preds

            # --- Logging (W&B + TensorBoard) ---
            log_dict = {
                "Loss/total": loss_value,
                "Loss_Physics/PDE_cryst": loss_dict["pde_cryst_loss"],
                "Loss_Physics/PDE_wm": loss_dict["pde_wm_loss"],
                "Loss_Physics/Mass_cryst": loss_dict["mass_cryst_loss"],
                "Loss_Physics/Mass_wm": loss_dict["mass_wm_loss"],
                "Loss_Physics/BC": loss_dict["bc_nLmax_loss"],
                "step": global_step,
                "epoch": epoch,
            }

            if cfg.logging.use_wandb:
                wandb.log(log_dict)
            else:
                for key, val in log_dict.items():
                    if key not in ["step", "epoch"]:
                        writer.add_scalar(key, val, global_step)

            # --- Optional: Log concentration slice vs true data ---
            # Pick a representative time near the middle of the batch
            # At end of epoch
            with torch.no_grad():
                t_all_norm = data["t"] / cfg.physics.t_scale
                c_pred_all, _ = conc_net(t_all_norm)
                c_pred_all = (c_pred_all * c_scale).cpu().numpy()
                c_true_all = data["c"].cpu().numpy()

                wandb.log({
                    "validation/concentration": wandb.plot.line_series(
                        xs=data["t"].cpu().numpy(),
                        ys=[c_pred_all, c_true_all],
                        keys=["pred", "true"],
                        title="Full Validation: c(t)",
                        xname="Time [s]"
                    ),
                    "validation/MAE": np.mean(np.abs(c_pred_all - c_true_all)),
                    "epoch": epoch
                })

            pbar.set_postfix({"loss": f"{loss_value:.3e}"})
            
            # Periodic memory cleanup (every 10 batches)
            if num_batches % 20 == 0:
                cleanup_memory()

        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.3e}")
        
        # Cleanup memory at end of epoch
        cleanup_memory()

        # Save checkpoint
        ckpt_path = f"pinn_pbm_checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "csd_state_dict": csd_net.state_dict(),
            "conc_state_dict": conc_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
            "L_grid": L_grid_phys.cpu(),
        }, ckpt_path)

        if cfg.logging.use_wandb:
            wandb.save(ckpt_path)

        # --- End-of-Epoch Evaluation: Plot CSD at representative times ---
        csd_net.eval()
        conc_net.eval()
        with torch.no_grad():
            eval_times = torch.linspace(0, cfg.physics.total_time, 5).to(device)
            for t_eval in eval_times:
                t_eval_norm = (t_eval / cfg.physics.t_scale).to(device)
                L_eval_norm = (L_grid_phys / cfg.physics.L_scale).to(device)
                n_c_hat_eval, _ = csd_net(
                    t_eval_norm.repeat(cfg.physics.n_L_grid),
                    L_eval_norm
                )
                n_c_eval = (n_c_hat_eval * cfg.physics.n_scale).cpu().numpy()

                if cfg.logging.use_wandb:
                    wandb.log({
                        f"CSD_cryst/t_{int(t_eval.item())}s": wandb.Histogram(n_c_eval),
                        "step": global_step
                    })
                else:
                    writer.add_histogram(f"CSD_cryst/t_{int(t_eval.item())}s", n_c_eval, global_step)

        csd_net.train()
        conc_net.train()

    writer.close()

    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()