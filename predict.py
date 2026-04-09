# ==================================================
# PREDICTION ENGINE (FINAL STABLE VERSION)
# ==================================================

import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import from_smiles

# ---------------------------------------------------------------
# 🔥 PATH SETUP (IMPORTANT FOR STREAMLIT)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------
# 🔥 LOAD FILES
# ---------------------------------------------------------------
feature_cols = joblib.load(os.path.join(BASE_DIR, "featuresG.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scalerG.pkl"))

device = torch.device("cpu")

# ---------------------------------------------------------------
# 🔥 REMOVE COLUMNS (SAME AS TRAINING)
# ---------------------------------------------------------------
remove_cols = [
    'solute_BalabanJ','solute_BertzCT','solute_Chi0','solute_Chi1',
    'solvent_BalabanJ','solvent_BertzCT','solvent_Chi0','solvent_Chi1',
    'solvent_Kappa1','solvent_Kappa2',
    'diff_BalabanJ','prod_BalabanJ','ratio_BalabanJ',
    'diff_BertzCT','prod_BertzCT','ratio_BertzCT',
    'diff_Chi0','prod_Chi0','ratio_Chi0',
    'diff_Chi1','prod_Chi1','ratio_Chi1',
    'diff_Kappa1','prod_Kappa1','ratio_Kappa1',
    'diff_Kappa2','prod_Kappa2','ratio_Kappa2'
]

# ---------------------------------------------------------------
# 🔥 MODEL DEFINITIONS
# ---------------------------------------------------------------
class MolEncoderGAT(torch.nn.Module):
    def __init__(self, in_dim=9, hidden=128):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=4)
        self.conv2 = GATConv(hidden * 4, hidden)

    def forward(self, data):
        x = F.elu(self.conv1(data.x.float(), data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        return global_mean_pool(x, data.batch)


class CrossAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, s, v):
        s = s.unsqueeze(1)
        v = v.unsqueeze(1)
        out, _ = self.attn(s, v, v)
        return out.squeeze(1)


class SolvGATNet(torch.nn.Module):
    def __init__(self, desc_dim):
        super().__init__()
        self.solute = MolEncoderGAT()
        self.solvent = MolEncoderGAT()
        self.cross = CrossAttention(128)

        self.desc_net = torch.nn.Sequential(
            torch.nn.Linear(desc_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        )

        self.mlp = torch.nn.Linear(128 + 64, 1)

    def forward(self, s, v, d):
        s_emb = self.solute(s)
        v_emb = self.solvent(v)
        g = self.cross(s_emb, v_emb)
        d = self.desc_net(d)
        return self.mlp(torch.cat([g, d], dim=1))


# ---------------------------------------------------------------
# 🔥 LOAD MODEL
# ---------------------------------------------------------------
model = SolvGATNet(desc_dim=len(feature_cols))
model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "best_modelG.pth"), map_location=device)
)
model.eval()


# ---------------------------------------------------------------
# 🔥 DESCRIPTOR FUNCTION (ALIGNED)
# ---------------------------------------------------------------
def compute_descriptors(solute_smiles, solvent_smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd

    def get_desc(smiles, prefix):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            f"{prefix}_MolWt": Descriptors.MolWt(mol),
            f"{prefix}_LogP": Descriptors.MolLogP(mol),
            f"{prefix}_TPSA": Descriptors.TPSA(mol),
            f"{prefix}_HBD": Descriptors.NumHDonors(mol),
            f"{prefix}_HBA": Descriptors.NumHAcceptors(mol),
        }

    # generate descriptors
    s = get_desc(solute_smiles, "solute")
    v = get_desc(solvent_smiles, "solvent")

    desc_dict = {**s, **v}

    df = pd.DataFrame([desc_dict])

    # -----------------------------------------------------------
    # 🔥 REMOVE COLUMNS (same as training)
    # -----------------------------------------------------------
    df = df.drop(columns=[c for c in remove_cols if c in df.columns], errors="ignore")

    # -----------------------------------------------------------
    # 🔥 ALIGN WITH TRAIN FEATURES
    # -----------------------------------------------------------
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols]  # correct order

    return df.values.astype(np.float32)[0]


# ---------------------------------------------------------------
# 🔥 PREDICT FUNCTION
# ---------------------------------------------------------------
def predict(solute, solvent):
    try:
        # graphs
        s_graph = from_smiles(solute)
        v_graph = from_smiles(solvent)

        s_graph.batch = torch.zeros(s_graph.num_nodes, dtype=torch.long)
        v_graph.batch = torch.zeros(v_graph.num_nodes, dtype=torch.long)

        # descriptors
        desc = compute_descriptors(solute, solvent)

        # scaling
        desc = scaler.transform([desc])[0]
        desc = torch.tensor(desc, dtype=torch.float).unsqueeze(0)

        # prediction
        with torch.no_grad():
            out = model(s_graph, v_graph, desc)

        return float(out.item())

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------
# 🔥 TEST
# ---------------------------------------------------------------
if __name__ == "__main__":
    solute = input("Enter solute SMILES: ")
    solvent = input("Enter solvent SMILES: ")

    result = predict(solute, solvent)
    print(result)
