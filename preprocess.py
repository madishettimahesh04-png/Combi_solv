
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

# ==========================================================
# RDKit Descriptor Names
# ==========================================================
descriptor_names = [

    name for name, _ in
    Descriptors._descList
]

# ==========================================================
# COMPUTE DESCRIPTORS
# ==========================================================
def compute_descriptors(
    mol,
    prefix
):

    values = {}

    for name, func in Descriptors._descList:

        try:

            values[
                f"{prefix}_{name}"
            ] = func(mol)

        except:

            values[
                f"{prefix}_{name}"
            ] = 0.0

    return values

# ==========================================================
# BUILD FEATURE VECTOR
# ==========================================================
def build_descriptors(

    solute_smiles,
    solvent_smiles
):

    solute = Chem.MolFromSmiles(
        solute_smiles
    )

    solvent = Chem.MolFromSmiles(
        solvent_smiles
    )

    # ------------------------------------------------------
    # Solute descriptors
    # ------------------------------------------------------
    solute_desc = compute_descriptors(
        solute,
        "solute"
    )

    # ------------------------------------------------------
    # Solvent descriptors
    # ------------------------------------------------------
    solvent_desc = compute_descriptors(
        solvent,
        "solvent"
    )

    # ------------------------------------------------------
    # Combine
    # ------------------------------------------------------
    features = {}

    features.update(solute_desc)

    features.update(solvent_desc)

    # ------------------------------------------------------
    # Difference descriptors
    # ------------------------------------------------------
    for name in descriptor_names:

        s1 = solute_desc[
            f"solute_{name}"
        ]

        s2 = solvent_desc[
            f"solvent_{name}"
        ]

        features[
            f"diff_{name}"
        ] = abs(s1 - s2)

    # ------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------
    df = pd.DataFrame([features])

    # ------------------------------------------------------
    # Clean
    # ------------------------------------------------------
    df = df.replace(
        [np.inf, -np.inf],
        0
    )

    df = df.fillna(0)

    return df
