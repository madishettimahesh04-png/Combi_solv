
import streamlit as st

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

from predict import predict_deltaG

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(

    page_title="Hybrid GNN DeltaG Predictor",

    layout="wide"
)

st.title(
    "Hybrid GNN ΔG Predictor"
)

# ==========================================================
# INPUTS
# ==========================================================
col1, col2 = st.columns(2)

with col1:

    solute_smiles = st.text_input(

        "Solute SMILES",

        value="CCO"
    )

with col2:

    solvent_smiles = st.text_input(

        "Solvent SMILES",

        value="O"
    )

# ==========================================================
# MOLECULE DISPLAY
# ==========================================================
col3, col4 = st.columns(2)

with col3:

    st.subheader("Solute")

    mol1 = Chem.MolFromSmiles(
        solute_smiles
    )

    if mol1:

        st.image(
            MolToImage(mol1)
        )

with col4:

    st.subheader("Solvent")

    mol2 = Chem.MolFromSmiles(
        solvent_smiles
    )

    if mol2:

        st.image(
            MolToImage(mol2)
        )

# ==========================================================
# PREDICTION BUTTON
# ==========================================================
if st.button(
    "Predict ΔG"
):

    try:

        prediction = predict_deltaG(

            solute_smiles,

            solvent_smiles
        )

        st.success(

            f"Predicted ΔG = "
            f"{prediction:.4f}"
        )

    except Exception as e:

        st.error(str(e))
