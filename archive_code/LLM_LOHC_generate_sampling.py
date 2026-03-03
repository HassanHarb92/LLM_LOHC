import requests
import pandas as pd
import json
import joblib
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 🔹 Adjustable Parameters
INITIAL_SET_SIZE = 5        # Number of molecules taken from CSV (Smart Sampled)
MAX_ITERATIONS = 10         # Maximum number of iterations to generate new SMILES
TARGET_COUNT = 5            # Desired number of final new LOHC molecules
NEW_LOHC_BATCH_SIZE = 5     # Number of new LOHCs requested per API call

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
SMILES_CSV_PATH = "Best_from_paper.csv"
OUTPUT_CSV_PATH = "Argo_LOHC_generated_test.csv"

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def compute_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """Generates Morgan fingerprint using the updated RDKit method."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return np.array(generator.GetFingerprint(mol))

def extract_smiles(text):
    """Extracts potential SMILES strings using regex and validates them."""
    smiles_pattern = r"[A-Za-z0-9@\[\]()+-=#$]+"
    candidates = re.findall(smiles_pattern, text)
    return [s for s in candidates if is_valid_smiles(s)]

def smart_sample_smiles(csv_path, sample_size):
    """Selects a diverse sample of SMILES using K-Means clustering."""
    try:
        df = pd.read_csv(csv_path)
        smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"📂 Loaded {len(smiles_list)} molecules from CSV.")

        # Compute fingerprints
        fingerprints = []
        valid_smiles = []
        for smiles in smiles_list:
            fp = compute_morgan_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_smiles.append(smiles)

        if len(valid_smiles) < sample_size:
            print("⚠️ Not enough diverse SMILES, returning all valid ones.")
            return valid_smiles  # If fewer than needed, return all valid ones

        fingerprints = np.array(fingerprints)

        # Reduce dimensionality for clustering (PCA)
        pca = PCA(n_components=10)
        reduced_fps = pca.fit_transform(fingerprints)

        # Cluster using K-Means
        kmeans = KMeans(n_clusters=sample_size, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(reduced_fps)

        # Pick one representative from each cluster
        sampled_smiles = []
        for cluster_id in range(sample_size):
            indices = np.where(clusters == cluster_id)[0]
            sampled_smiles.append(valid_smiles[indices[0]])  # Pick the first molecule in each cluster

        print(f"✅ Selected {len(sampled_smiles)} diverse SMILES for initial set.")
        return sampled_smiles

    except Exception as e:
        print(f"❌ Error in smart sampling: {e}")
        return []

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

        # 🔹 Fix: Strip `json` formatting (remove triple backticks)
        raw_output = raw_output.strip("```json").strip("```").strip()

        try:
            json_output = eval(raw_output)  
            print(f"✅ Received {len(json_output.get('SMILES', []))} new SMILES.")
            return json_output.get("SMILES", [])
        except Exception:
            return extract_smiles(raw_output)

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return []

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
        filtered_smiles = [s for s in new_smiles if s not in initial_smiles_set]

        iterative_set.update(filtered_smiles)
        final_results.extend(filtered_smiles)

        if not filtered_smiles:
            print("❌ No valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results

# Main Execution
if __name__ == "__main__":
    initial_smiles = smart_sample_smiles(SMILES_CSV_PATH, INITIAL_SET_SIZE)

    if not initial_smiles:
        print("❌ No valid initial SMILES found. Exiting.")
        exit()

    print(f"🔹 Initial selected SMILES: {initial_smiles}")

    final_lohc_set = iterative_generation(initial_smiles)

    if final_lohc_set:
        df_out = pd.DataFrame({"Generated SMILES": final_lohc_set})
        df_out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ SMILES exported successfully to '{OUTPUT_CSV_PATH}'")
    else:
        print("❌ No valid LOHC SMILES generated.")

