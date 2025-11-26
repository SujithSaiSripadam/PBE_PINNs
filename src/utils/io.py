import pandas as pd
import torch

def load_data(csv_path, nrows=None, device="cpu", dtype=torch.float32):
    if nrows == None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, nrows=nrows)
        
    return {
        "t": torch.tensor(df["t"].values, device=device, dtype=dtype),
        "T": torch.tensor(df["T"].values, device=device, dtype=dtype),
        "F": torch.tensor(df["F"].values, device=device, dtype=dtype),
        "N": torch.tensor(df["N"].values, device=device, dtype=dtype),
        "c": torch.tensor(df["c"].values, device=device, dtype=dtype),
        "cwm": torch.tensor(df["cwm"].values, device=device, dtype=dtype),
    }