import streamlit as st
from predict import predict

st.title("🧪 Solvation Energy Predictor")

solute = st.text_input("Solute SMILES", "CCO")
solvent = st.text_input("Solvent SMILES", "O")

if st.button("Predict"):
    result = predict(solute, solvent)

    if isinstance(result, float):
        st.success(f"Predicted Energy: {result:.4f} kcal/mol")
    else:
        st.error(result)
