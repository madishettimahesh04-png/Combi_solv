import streamlit as st
from predict import predict

st.set_page_config(page_title="Solvation Predictor")

st.title("🧪 Solvation Energy Predictor")

solute = st.text_input("Solute SMILES", "CCO")
solvent = st.text_input("Solvent SMILES", "O")

if st.button("Predict"):
    result = predict(solute, solvent)
    st.success(f"Predicted Energy: {result:.4f} kcal/mol")
