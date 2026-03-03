import requests
import pandas as pd
import json
import joblib
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import time
from datetime import timedelta


# Add Start time

start_time = time.time()

# 🔹 Adjustable Parameters
INITIAL_SET_SIZE = 28        # Number of molecules taken from CSV
MAX_ITERATIONS = 30         # Maximum number of iterations to generate new SMILES
TARGET_COUNT = 500           # Desired number of final new LOHC molecules
NEW_LOHC_BATCH_SIZE = 10     # Number of new LOHCs requested per API call

# Read API URL from `api.txt`
try:
    with open("api.txt", "r") as file:
        API_URL = file.read().strip()
        print(f"🔹 API URL Loaded: {API_URL}")
except FileNotFoundError:
    print("❌ Error: `api.txt` not found. Please create the file and add the API URL.")
    exit()

# API Configuration
HEADERS = {"Content-Type": "application/json"}

# Load the trained RF model
rf_model = joblib.load("QM9-LOHC-RF.joblib")

# File paths
SMILES_CSV_PATH = "expt_31.csv"
OUTPUT_CSV_PATH = "Argo_LOHC_generated_from_expt.csv"

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid and returns its canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)  # Convert to canonical SMILES

def hydrogenate_smiles(smiles):
    """Generates a hydrogenated counterpart of the given SMILES."""
    hydrogenated = smiles.replace("=", "").replace("#", "")
    hydrogenated = ''.join([char.upper() if char in 'cons' else char for char in hydrogenated])
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
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

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
    valid_smiles = []
    
    for s in candidates:
        canonical_s = is_valid_smiles(s)
        if canonical_s:
            valid_smiles.append(canonical_s)
    
    return valid_smiles

def generate_new_smiles(initial_smiles):
    """Generates new LOHC SMILES using the fixed API request format."""
    print(f"🔹 Sending API request to generate {NEW_LOHC_BATCH_SIZE} new SMILES...")

    prompt = f"""
    You are an expert in molecular design, specializing in Liquid Organic Hydrogen Carriers (LOHCs).
    
    The user provided these known LOHC SMILES:
    {', '.join(initial_smiles)}

    Your task is to generate exactly {NEW_LOHC_BATCH_SIZE} novel LOHC SMILES strings in a structured JSON format:
    {{"SMILES": ["SMILES1", "SMILES2", "SMILES3", ..., "SMILES{NEW_LOHC_BATCH_SIZE}"]}}

    Ensure that the new SMILES are chemically valid, unique, and not already in the provided list.
    Do not include any additional text or explanations. Respond only with the JSON structure.
    """

    payload = {
        "user": "hharb",
        "model": "gpto1preview",
        "system": "You are a super smart and helpful AI for materials science research.",
        "prompt": [prompt],
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
            json_output = eval(raw_output)  
            print(f"✅ Received {len(json_output.get('SMILES', []))} new SMILES.")
            return extract_smiles(json_output.get("SMILES", []))
        except Exception:
            return extract_smiles(raw_output)

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return []

def filter_and_evaluate_smiles(smiles_list, initial_smiles_set):
    """Filters generated SMILES based on validity, H2 %, and ΔH predictions, ensuring they are not in the initial set."""
    valid_smiles = []
    results = []

    print("🔹 Filtering and evaluating new SMILES...")

    for smiles in smiles_list:
        if smiles in initial_smiles_set:
            print(f"❌ {smiles} is in the initial dataset, skipping.")
            continue

        h2_weight = calculate_hydrogen_weight(smiles)
        if h2_weight is None:
            print(f"❌ {smiles} failed H₂ weight requirement.")
            continue

        delta_h = predict_delta_h(smiles)
        if delta_h is None or not (40 <= delta_h <= 70):
            print(f"❌ {smiles} failed RF ΔH prediction (got {delta_h}).")
            continue

        print(f"✅ {smiles} passed all filters. ΔH: {delta_h}")
        valid_smiles.append(smiles)
        results.append({"SMILES": smiles, "Predicted ΔH": delta_h})

    return valid_smiles, results

def iterative_generation(initial_set):
    """Generates LOHC SMILES iteratively until reaching TARGET_COUNT."""
    iterative_set = set()
    final_results = []
    initial_smiles_set = set(initial_set)

    for i in range(MAX_ITERATIONS):
        if len(iterative_set) >= TARGET_COUNT:
            break
        
        print(f"\n🔹 Iteration {i+1}/{MAX_ITERATIONS} - Current set size: {len(iterative_set)}.")

        new_smiles = generate_new_smiles(list(iterative_set) + list(initial_smiles_set))
        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles, initial_smiles_set)

        iterative_set.update(filtered_smiles)
        final_results.extend(results)

        if not filtered_smiles:
            print("❌ No valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results

# Main Execution
if __name__ == "__main__":
    try:
        df = pd.read_csv(SMILES_CSV_PATH)
        initial_smiles = df.iloc[:INITIAL_SET_SIZE, 0].dropna().astype(str).tolist()
        initial_smiles = [is_valid_smiles(s) for s in initial_smiles if is_valid_smiles(s)]
        print(f"📂 Loaded {len(initial_smiles)} initial SMILES from CSV.")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        exit()

    final_lohc_set = iterative_generation(initial_smiles)
 
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Compute elapsed time
    formatted_time = str(timedelta(seconds=elapsed_time)).split(".")[0]
    print(f"Execution time: {formatted_time}")


    if final_lohc_set:
        df_out = pd.DataFrame(final_lohc_set)
        df_out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ SMILES exported successfully to '{OUTPUT_CSV_PATH}'")
    else:
        print("❌ No valid LOHC SMILES generated.")

