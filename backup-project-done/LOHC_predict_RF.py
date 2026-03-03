from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
import numpy as np
import joblib  # Import joblib for loading the model

# Load the trained model
model = joblib.load("QM9-LOHC-RF.joblib")

def compute_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    generator = GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp)

# Example list of new SMILES strings
new_smiles = ['C1=CC=CC=C1', 'CC(=O)O']  # Replace with your actual SMILES strings

# Convert these SMILES strings to Morgan fingerprints
new_morgan_fps = [compute_morgan_fingerprint(smile) for smile in new_smiles]

# Convert to numpy array for the model
new_morgan_fps = np.array(new_morgan_fps)

# Predict delta_H values using the trained model
new_predictions = model.predict(new_morgan_fps)

print(new_predictions)

