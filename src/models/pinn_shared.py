import torch
import torch.nn as nn


class PINN_SHARED(nn.Module):
    """Single shared network with two branches:
    - concentration branch (t, T, F, N) -> c_c_hat, c_wm_hat
    - number-density branch: uses shared features + L -> n_c_hat, n_wm_hat
    The object is callable with either signature to preserve backward compatibility:
      shared(t_norm, L_norm, T_norm, F_norm, N_norm) -> n_c_hat, n_wm_hat
      shared(t_norm, T_norm, F_norm, N_norm) -> c_c_hat, c_wm_hat
    """

    def __init__(self, hidden_dim=128, activation="SiLU", num_layers=5):
        super().__init__()
        act_fn = getattr(nn, activation)()

        # Shared trunk operates on (t, T, F, N)
        trunk_layers = []
        trunk_layers.append(nn.Linear(4, hidden_dim))
        trunk_layers.append(act_fn)
        for _ in range(num_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(act_fn)
        self.trunk = nn.Sequential(*trunk_layers)

        # Concentration head: from trunk features -> 2 outputs
        conc_layers = []
        conc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        conc_layers.append(act_fn)
        conc_layers.append(nn.Linear(hidden_dim, 2))
        self.conc_head = nn.Sequential(*conc_layers)

        # CSD head: takes trunk features concatenated with L_norm (1 dim)
        csd_layers = []
        csd_layers.append(nn.Linear(hidden_dim + 1, hidden_dim))
        csd_layers.append(act_fn)
        for _ in range(max(1, num_layers - 2)):
            csd_layers.append(nn.Linear(hidden_dim, hidden_dim))
            csd_layers.append(act_fn)
        csd_layers.append(nn.Linear(hidden_dim, 2))
        self.csd_head = nn.Sequential(*csd_layers)

        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, *args):
        # Support two call signatures for backward compatibility
        # If five args: (t_norm, L_norm, T_norm, F_norm, N_norm) -> csd outputs
        # If four args: (t_norm, T_norm, F_norm, N_norm) -> conc outputs
        if len(args) == 5:
            t_norm, L_norm, T_norm, F_norm, N_norm = args
            # build trunk input: (t, T, F, N)
            t = t_norm.view(-1, 1)
            T = T_norm.view(-1, 1)
            F = F_norm.view(-1, 1)
            N = N_norm.view(-1, 1)
            trunk_in = torch.cat([t, T, F, N], dim=1)
            feats = self.trunk(trunk_in)

            # csd head: concat L
            L = L_norm.view(-1, 1)
            csd_in = torch.cat([feats, L], dim=1)
            out = self.csd_head(csd_in)
            out = self.softplus(out)
            return out[:, 0], out[:, 1]

        elif len(args) == 4:
            t_norm, T_norm, F_norm, N_norm = args
            t = t_norm.view(-1, 1)
            T = T_norm.view(-1, 1)
            F = F_norm.view(-1, 1)
            N = N_norm.view(-1, 1)
            trunk_in = torch.cat([t, T, F, N], dim=1)
            feats = self.trunk(trunk_in)
            out = self.conc_head(feats)
            out = self.softplus(out)
            return out[:, 0], out[:, 1]

        else:
            raise ValueError("PINN_SHARED.forward expects 4 (conc) or 5 (csd) tensors")