
import torch
import joblib
import pandas as pd

from torch_geometric.utils import from_smiles
from torch_geometric.data import Batch

from model import Model
from preprocess import build_descriptors

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# LOAD FILES
# ==========================================================
pipeline = joblib.load(
    "pipeline.pkl"
)

feature_order = joblib.load(
    "feature_order.pkl"
)

config = joblib.load(
    "model_config.pkl"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
model = Model(

    config["input_dim"],

    config["hidden_dim"],

    config["heads"],

    config["dropout"],

    config["desc_hidden"]
)

model.load_state_dict(

    torch.load(
        "best_model.pt",
        map_location=device
    )
)

model.to(device)

model.eval()

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_deltaG(

    solute_smiles,
    solvent_smiles
):

    # ------------------------------------------------------
    # DESCRIPTORS
    # ------------------------------------------------------
    desc_df = build_descriptors(

        solute_smiles,
        solvent_smiles
    )

    # ------------------------------------------------------
    # REMOVE CORRELATED FEATURES
    # ------------------------------------------------------
    desc_df = desc_df.drop(

        columns=pipeline[
            "removed_corr_features"
        ],

        errors="ignore"
    )

    # ------------------------------------------------------
    # FEATURE ORDER
    # ------------------------------------------------------
    desc_df = desc_df.reindex(

        columns=feature_order,

        fill_value=0
    )

    # ------------------------------------------------------
    # SCALING
    # ------------------------------------------------------
    desc_scaled = pipeline[
        "scaler"
    ].transform(desc_df)

    desc_tensor = torch.tensor(

        desc_scaled,

        dtype=torch.float32
    ).to(device)

    # ------------------------------------------------------
    # GRAPH GENERATION
    # ------------------------------------------------------
    g1 = from_smiles(
        solute_smiles
    )

    g2 = from_smiles(
        solvent_smiles
    )

    g1.x = g1.x.float()

    g2.x = g2.x.float()

    g1 = Batch.from_data_list([g1])

    g2 = Batch.from_data_list([g2])

    g1 = g1.to(device)

    g2 = g2.to(device)

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------
    with torch.no_grad():

        pred = model(
            g1,
            g2,
            desc_tensor
        )

    return float(pred.item())
