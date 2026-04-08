# ==================================================
# PREDICTION ENGINE (DEPLOYMENT)
# ==================================================

import torch
import numpy as np
import joblib
from torch_geometric.utils import from_smiles

# -------------------------------
# LOAD FILES
# -------------------------------
device = torch.device("cpu")

model = SolvGATNet(desc_dim=len(joblib.load("features.pkl")))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")
selector = joblib.load("variance.pkl")
feature_cols = joblib.load("features.pkl")

# -------------------------------
# RDKit DESCRIPTORS
# -------------------------------
def compute_descriptors(solute_smiles, solvent_smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    def get_desc(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(50)

        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ], dtype=np.float32)

    s = get_desc(solute_smiles)
    v = get_desc(solvent_smiles)

    # combine (simple version)
    desc = np.concatenate([s, v])

    # pad to match training size
    if len(desc) < len(feature_cols):
        desc = np.pad(desc, (0, len(feature_cols) - len(desc)))

    return desc

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(solute, solvent):

    s_graph = from_smiles(solute)
    v_graph = from_smiles(solvent)

    s_graph.batch = torch.zeros(s_graph.num_nodes, dtype=torch.long)
    v_graph.batch = torch.zeros(v_graph.num_nodes, dtype=torch.long)

    desc = compute_descriptors(solute, solvent)

    # apply variance filter
    desc = selector.transform([desc])[0]

    # scale
    desc = scaler.transform([desc])[0]
    desc = torch.tensor(desc, dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        out = model(s_graph, v_graph, desc)

    return float(out.item())


# -------------------------------
# TEST RUN
# -------------------------------
if __name__ == "__main__":
    solute = input("Enter solute SMILES: ")
    solvent = input("Enter solvent SMILES: ")

    result = predict(solute, solvent)
    print(f"Prediction: {result:.4f} kcal/mol")
