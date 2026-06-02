
import streamlit as st

from predict import predict_deltaG

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(

    page_title="Hybrid GNN DeltaG Predictor",

    layout="centered"
)

# ==========================================================
# TITLE
# ==========================================================
st.title(
    "Hybrid GNN ΔG Predictor"
)

st.markdown(
    "Predict ΔG using Solute and Solvent SMILES"
)

# ==========================================================
# INPUTS
# ==========================================================
solute_smiles = st.text_input(

    "Solute SMILES",

    value="CCO"
)

solvent_smiles = st.text_input(

    "Solvent SMILES",

    value="O"
)

# ==========================================================
# BUTTON
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

        st.error(

            f"Prediction failed: {str(e)}"
        )
