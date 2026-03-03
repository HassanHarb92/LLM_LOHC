import requests
import pandas as pd
import json
import joblib
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

# Load the trained RF model
rf_model = joblib.load("QM9-LOHC-RF.joblib")

# API Configuration (Based on Your Working Example)
API_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"  # Use your working endpoint here
HEADERS = {"Content-Type": "application/json"}

# File paths
SMILES_CSV_PATH = "Best_from_paper.csv"
OUTPUT_CSV_PATH = "Argo_LOHC_generated.csv"

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def old_hydrogenate_smiles(smiles):
    """Hydrogenates a SMILES string by removing double/triple bonds and uppercasing specific atoms."""
    return smiles.replace("=", "").replace("#", "").upper()

def hydrogenate_smiles(smiles):
    """Generates a hydrogenated counterpart of the given SMILES."""
    hydrogenated = smiles.replace("=", "").replace("#", "")
    hydrogenated = ''.join([char.upper() if char in 'cons' else char for char in hydrogenated])
    return hydrogenated
    """Generates a hydrogenated counterpart of the given SMILES."""
    hydrogenated = smiles.replace("=", "").replace("#", "")
    hydrogenated = ''.join([char.upper() if char in 'cons' else char for char in hydrogenated])
    return hydrogenated
    """Generates a hydrogenated counterpart of the given SMILES."""
    hydrogenated = smiles.replace("=", "").replace("#", "").upper()
    return hydrogenated



def calculate_hydrogen_weight(smiles):
    """Calculates the hydrogen weight percentage of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw_original = Descriptors.MolWt(mol)
    hydrogenated_smiles = hydrogenate_smiles(smiles)
    mol_hydrogenated = Chem.MolFromSmiles(hydrogenated_smiles)

    if mol_hydrogenated is None:
        return None

    mw_hydrogenated = Descriptors.MolWt(mol_hydrogenated)
    h2_percent = (mw_hydrogenated - mw_original) * 100 / mw_hydrogenated

    return h2_percent if h2_percent >= 5.5 else None

def compute_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """Generates Morgan fingerprint for a given SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return np.array(generator.GetFingerprint(mol))

def predict_delta_h(smiles):
    """Predicts ΔH using the trained RF model."""
    fingerprint = compute_morgan_fingerprint(smiles)
    if fingerprint is None:
        return None
    fingerprint = np.array(fingerprint).reshape(1, -1)  # Ensure correct input shape
    return rf_model.predict(fingerprint)[0]

def extract_smiles(text):
    """Extracts potential SMILES strings using regex and validates them."""
    smiles_pattern = r"[A-Za-z0-9@\[\]()+-=#$]+"
    candidates = re.findall(smiles_pattern, text)
    return [s for s in candidates if is_valid_smiles(s)]

def generate_new_smiles(initial_smiles, batch_size=50):
    """Generates new LOHC SMILES using the fixed API request format."""
    prompt = f"""
    You are an expert in molecular design, specializing in Liquid Organic Hydrogen Carriers (LOHCs).
    
    The user provided these known LOHC SMILES:
    {', '.join(initial_smiles)}

    Your task is to generate exactly {batch_size} novel LOHC SMILES strings in a structured JSON format:
    {{"SMILES": ["SMILES1", "SMILES2", "SMILES3", ..., "SMILES{batch_size}"]}}

    Ensure that the new SMILES are chemically valid and at least as good or better than the provided ones.
    Do not include any additional text or explanations. Respond only with the JSON structure.
    """

    payload = {
        "user": "hharb",  # Adjust based on your working API
        "model": "gpto1preview",
        "system": "You are a super smart and helpful AI for materials science research.",
        "prompt": [prompt],  # Use list format based on working example
        "stop": [],
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 2000,
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        response_json = response.json()
        raw_output = response_json.get("response", "")

        try:
            json_output = eval(raw_output)  # Convert JSON-like string to dict
            return json_output.get("SMILES", [])
        except Exception:
            return extract_smiles(raw_output)

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []

def filter_and_evaluate_smiles(smiles_list):
    """Filters generated SMILES based on validity, H2 %, and ΔH predictions."""
    valid_smiles = []
    results = []

    for smiles in smiles_list:
        if not is_valid_smiles(smiles):
            continue

        h2_weight = calculate_hydrogen_weight(smiles)
        if h2_weight is None:
            continue

        delta_h = predict_delta_h(smiles)
        if delta_h is None or not (40 <= delta_h <= 70):
            continue

        valid_smiles.append(smiles)
        results.append({"SMILES": smiles, "Predicted ΔH": delta_h})

    return valid_smiles, results

def iterative_generation(initial_set, target_count=100):
    """Generates LOHC SMILES iteratively until reaching target_count."""
    iterative_set = set(initial_set)
    final_results = []

    while len(iterative_set) < target_count:
        print(f"Current LOHC set size: {len(iterative_set)}. Generating new candidates...")
        print("Current set: ",iterative_set)
        # Generate 50 new candidate SMILES
        new_smiles = generate_new_smiles(list(iterative_set))
        
        # Filter and evaluate the new candidates
        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles)

        # Merge the filtered set with existing set
        iterative_set.update(filtered_smiles)
        final_results.extend(results)

        if not filtered_smiles:
            print("No valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results

# Main Execution
if __name__ == "__main__":
    # Read initial SMILES from CSV
    try:
        df = pd.read_csv(SMILES_CSV_PATH)
        initial_smiles = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit()

    # Start iterative generation process
    final_lohc_set = iterative_generation(initial_smiles)

    # Save final results
    if final_lohc_set:
        df_out = pd.DataFrame(final_lohc_set)
        df_out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"SMILES exported successfully to '{OUTPUT_CSV_PATH}'")
    else:
        print("No valid LOHC SMILES generated.")

