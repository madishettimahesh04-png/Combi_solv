# ==================================================
# PREDICTION ENGINE (FINAL FIXED VERSION)
# ==================================================

import torch
import torch.nn.functional as F
import numpy as np
import joblib

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import from_smiles

# ---------------------------------------------------------------
# 🔥 MODEL DEFINITIONS (MANDATORY)
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
# LOAD FILES
# ---------------------------------------------------------------
device = torch.device("cpu")

feature_cols = joblib.load("features.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("variance.pkl")

model = SolvGATNet(desc_dim=len(feature_cols))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


# ---------------------------------------------------------------
# 🔥 DESCRIPTOR FUNCTION (IMPROVED)
# ---------------------------------------------------------------
def compute_descriptors(solute_smiles, solvent_smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    def get_desc(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(10)

        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol),
            Descriptors.NHOHCount(mol),
        ], dtype=np.float32)

    s = get_desc(solute_smiles)
    v = get_desc(solvent_smiles)

    # combine
    desc = np.concatenate([s, v])

    # pad safely
    if len(desc) < len(feature_cols):
        desc = np.pad(desc, (0, len(feature_cols) - len(desc)))
    else:
        desc = desc[:len(feature_cols)]

    return desc


# ---------------------------------------------------------------
# PREDICTION FUNCTION
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

        # variance filter
        desc = selector.transform([desc])[0]

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
# TEST RUN
# ---------------------------------------------------------------
if __name__ == "__main__":
    solute = input("Enter solute SMILES: ")
    solvent = input("Enter solvent SMILES: ")

    result = predict(solute, solvent)
    print(f"Prediction: {result}")
