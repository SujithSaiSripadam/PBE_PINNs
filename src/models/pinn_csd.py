import torch
import torch.nn as nn

class PINN_CSD(nn.Module):
    def __init__(self, hidden_dim=128, activation="SiLU", num_layers=5):
        super().__init__()
        act_fn = getattr(nn, activation)()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(act_fn)
        
        for _ in range(num_layers - 1):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
            
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, t_norm, L_norm):
        x = torch.cat([t_norm.view(-1,1), L_norm.view(-1,1)], dim=1)
        out = self.net(x)
        out = self.softplus(out)
        return out[:,0], out[:,1]  # n_c_hat, n_wm_hat