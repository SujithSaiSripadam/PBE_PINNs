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
from scipy.interpolate import interp1d
import sys

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
    # Parse command-line arguments for checkpoint loading
    checkpoint_path = None
    for arg in sys.argv:
        if arg.startswith("--ckpt="):
            checkpoint_path = arg.split("=", 1)[1]
        elif arg.startswith("--checkpoint="):
            checkpoint_path = arg.split("=", 1)[1]
    
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
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            csd_net.load_state_dict(ckpt["csd_state_dict"])
            conc_net.load_state_dict(ckpt["conc_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        else:
            print(f"\nWarning: Checkpoint path does not exist: {checkpoint_path}")
            print("Starting fresh training...")
    else:
        print("\nNo checkpoint provided. Starting fresh training...")

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

    # Learning rate scheduler (cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.n_epochs,
        eta_min=1e-7
    )

    # Logging
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name= hydra.core.hydra_config.HydraConfig.get().job.name,
            settings=wandb.Settings(show_warnings = False)
        )
        wandb.watch((csd_net, conc_net), log="all", log_freq=cfg.logging.log_freq)

    # ---------------------------
    # Training Loop - AdamW Stage
    # ---------------------------
    global_step = 0
    writer = SummaryWriter(log_dir="runs/" + hydra.core.hydra_config.HydraConfig.get().job.name)

    if cfg.training.use_adamw:
        print("\n" + "="*60)
        print("Starting AdamW Training Stage")
        print("="*60)
        
        for epoch in range(start_epoch, cfg.training.n_epochs):
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
                #del t_b, L_b, T_b, F_b, N_b, loss_phys, preds

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
                """
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
                if num_batches % 1 == 0:
                    cleanup_memory()
                """
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.3e}")
            
            # Step scheduler at end of epoch
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning Rate: {current_lr:.3e}")
            
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
                t_data_cpu = data["t"].cpu().numpy()
                T_data_cpu = data["T"].cpu().numpy()
                F_data_cpu = data["F"].cpu().numpy()
                N_data_cpu = data["N"].cpu().numpy()

                T_interp = interp1d(t_data_cpu, T_data_cpu, kind='linear', fill_value="extrapolate")
                F_interp = interp1d(t_data_cpu, F_data_cpu, kind='linear', fill_value="extrapolate")
                N_interp = interp1d(t_data_cpu, N_data_cpu, kind='linear', fill_value="extrapolate")

                eval_times = torch.linspace(0, cfg.physics.total_time, 5).cpu().numpy()
                
                for t_eval in eval_times:
                    # Get T, F, N at this time
                    T_eval = T_interp(t_eval).item()
                    F_eval = F_interp(t_eval).item()
                    N_eval = N_interp(t_eval).item()
                    
                    # Create tensors repeated over L grid
                    t_tensor = torch.full((cfg.physics.n_L_grid,), t_eval, device=device, dtype=torch.float32)
                    T_tensor = torch.full((cfg.physics.n_L_grid,), T_eval, device=device, dtype=torch.float32)
                    F_tensor = torch.full((cfg.physics.n_L_grid,), F_eval, device=device, dtype=torch.float32)
                    N_tensor = torch.full((cfg.physics.n_L_grid,), N_eval, device=device, dtype=torch.float32)
                    
                    # Normalize
                    t_norm = (t_tensor / cfg.physics.t_scale)
                    L_norm = (L_grid_phys / cfg.physics.L_scale)
                    T_norm = (T_tensor / cfg.physics.T_scale)
                    F_norm = (F_tensor / cfg.physics.F_scale)
                    N_norm = (N_tensor / cfg.physics.N_scale)
                    
                    # Forward pass
                    n_c_hat_eval, _ = csd_net(t_norm, L_norm, T_norm, F_norm, N_norm)
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
        
        print("\nAdamW Training Complete!")
        print(f"Final Loss: {avg_epoch_loss:.3e}")
    
    else:
        print("\n" + "="*60)
        print("AdamW Training Skipped (use_adamw=false)")
        print("="*60)

    # ---------------------------
    # L-BFGS Fine-tuning Stage (Optional)
    # ---------------------------
    if cfg.training.use_lbfgs:
        print("\n" + "="*60)
        print("Starting L-BFGS Fine-tuning Stage")
        print("="*60)
        
        csd_net.train()
        conc_net.train()
        
        # Create L-BFGS optimizer
        lbfgs_optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg.training.lbfgs_max_iter,
            line_search_fn='strong_wolfe'
        )
        
        lbfgs_global_step = global_step
        lbfgs_loss_history = []
        
        for lbfgs_epoch in range(cfg.training.lbfgs_epochs):
            lbfgs_epoch_loss = 0.0
            lbfgs_num_batches = 0
            
            pbar_lbfgs = tqdm(colloc_loader, desc=f"L-BFGS Step {lbfgs_epoch+1}/{cfg.training.lbfgs_epochs}", leave=False)
            
            for batch in pbar_lbfgs:
                t_b, L_b, T_b, F_b, N_b = [x.to(device) for x in batch]
                
                def closure():
                    lbfgs_optimizer.zero_grad()
                    loss_phys, loss_dict, preds = phys.compute_loss(csd_net, conc_net, t_b, L_b, T_b, F_b, N_b)
                    loss_phys.backward()
                    return loss_phys
                
                loss_value = lbfgs_optimizer.step(closure)
                
                # Accumulate
                lbfgs_epoch_loss += loss_value.item()
                lbfgs_num_batches += 1
                lbfgs_global_step += 1
                
                pbar_lbfgs.set_postfix({"loss": f"{loss_value.item():.3e}"})
                
                # Logging
                log_dict_lbfgs = {
                    "Loss/total_lbfgs": loss_value.item(),
                    "Loss_Physics/PDE_cryst": loss_dict["pde_cryst_loss"],
                    "Loss_Physics/PDE_wm": loss_dict["pde_wm_loss"],
                    "Loss_Physics/Mass_cryst": loss_dict["mass_cryst_loss"],
                    "Loss_Physics/Mass_wm": loss_dict["mass_wm_loss"],
                    "Loss_Physics/BC": loss_dict["bc_nLmax_loss"],
                    "lbfgs_step": lbfgs_global_step,
                    "lbfgs_epoch": lbfgs_epoch,
                }
                
                if cfg.logging.use_wandb:
                    wandb.log(log_dict_lbfgs)
                else:
                    for key, val in log_dict_lbfgs.items():
                        if key not in ["lbfgs_step", "lbfgs_epoch"]:
                            writer.add_scalar(key, val, lbfgs_global_step)
                
                cleanup_memory()
            
            # End of L-BFGS epoch
            avg_lbfgs_epoch_loss = lbfgs_epoch_loss / lbfgs_num_batches
            lbfgs_loss_history.append(avg_lbfgs_epoch_loss)
            print(f"L-BFGS Step {lbfgs_epoch+1} | Avg Loss: {avg_lbfgs_epoch_loss:.3e}")
            
            cleanup_memory()
            
            # Save checkpoint after each L-BFGS step
            ckpt_path_lbfgs = f"pinn_pbm_checkpoint_lbfgs_step_{lbfgs_epoch}.pt"
            torch.save({
                "epoch": cfg.training.n_epochs + lbfgs_epoch,
                "lbfgs_step": lbfgs_epoch,
                "csd_state_dict": csd_net.state_dict(),
                "conc_state_dict": conc_net.state_dict(),
                "optimizer_state_dict": lbfgs_optimizer.state_dict(),
                "loss": avg_lbfgs_epoch_loss,
                "L_grid": L_grid_phys.cpu(),
            }, ckpt_path_lbfgs)
            
            if cfg.logging.use_wandb:
                wandb.save(ckpt_path_lbfgs)
        
        print("L-BFGS Fine-tuning Complete")
        print(f"Final L-BFGS Loss: {lbfgs_loss_history[-1]:.3e}")

    writer.close()

    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    # Usage Examples:
    # ============================================
    # 1. Fresh start with AdamW + L-BFGS:
    #    python scripts/train.py
    #
    # 2. Fresh start with AdamW only:
    #    python scripts/train.py training.use_lbfgs=false
    #
    # 3. Fresh start with L-BFGS only (no AdamW):
    #    python scripts/train.py training.use_adamw=false
    #
    # 4. Load checkpoint and continue with AdamW + L-BFGS:
    #    python scripts/train.py --ckpt=pinn_pbm_checkpoint_epoch_50.pt
    #
    # 5. Load checkpoint and continue with L-BFGS only:
    #    python scripts/train.py --ckpt=pinn_pbm_checkpoint_epoch_50.pt training.use_adamw=false
    #
    # 6. Load checkpoint and run AdamW only:
    #    python scripts/train.py --ckpt=pinn_pbm_checkpoint_epoch_50.pt training.use_lbfgs=false
    # ============================================
    main()