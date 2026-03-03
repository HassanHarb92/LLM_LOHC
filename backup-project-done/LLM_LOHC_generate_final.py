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
import leruli
import sys

log_file = open("output_expt.log", "a")  # Append mode
sys.stdout = log_file
sys.stderr = log_file  # Capture errors too

print("This is logged!", flush=True)

# Ensure logging is written in real-time
sys.stdout.flush()
sys.stderr.flush()

# Add Start time

start_time = time.time()

# 🔹 Adjustable Parameters
INITIAL_SET_SIZE = 31        # Number of molecules taken from CSV
MAX_ITERATIONS = 10         # Maximum number of iterations to generate new SMILES
TARGET_COUNT = 200           # Desired number of final new LOHC molecules
NEW_LOHC_BATCH_SIZE = 30     # Number of new LOHCs requested per API call
MAX_LLM_ATTEMPTS = 10

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
OUTPUT_CSV_PATH = "Argo_LOHC_generated_dry_run_withMP_more_attempts_second_try_expt.csv"

def predict_melting_point(smiles):
    """Fetches the melting point for a given SMILES string using Leruli's SDK."""
    try:
        data_mp = leruli.graph_to_melting_point(smiles)
        return data_mp.get("mp", None)  # Extract melting point if available
    except Exception as e:
        print(f"Error fetching melting point for {smiles}: {e}")
        return 9999  # Return None if no value is found or an error occurs

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid and returns its canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Invalid SMILES detected: {smiles}")  # Log invalid SMILES
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

def old_extract_smiles(text):
    """Extracts potential SMILES strings using regex and validates them."""
    smiles_pattern = r"[A-Za-z0-9@\[\]()+-=#$]+"
    candidates = re.findall(smiles_pattern, text)
    valid_smiles = []
    
    for s in candidates:
        canonical_s = is_valid_smiles(s)
        if canonical_s:
            valid_smiles.append(canonical_s)
    
    return valid_smiles

def extract_smiles(text):
    """Extracts potential SMILES strings using regex and validates them.
       Handles both strings and lists as input.
    """
    smiles_pattern = r"[A-Za-z0-9@\[\]()+-=#$]+"

    # Ensure input is a single string before applying regex
    if isinstance(text, list):
        text = " ".join(text)  # Convert list to a single space-separated string

    candidates = re.findall(smiles_pattern, text)  # Apply regex
    valid_smiles = []

    for s in candidates:
        canonical_s = is_valid_smiles(s)
        if canonical_s:
            valid_smiles.append(canonical_s)

    return valid_smiles



def old_generate_new_smiles(initial_smiles):
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

        unique_smiles = list(set(received_smiles))  # Convert list to set and back to list
        print(f"✅ Received {len(unique_smiles)} unique SMILES after removing duplicates.")

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return []

def old_2_generate_new_smiles(initial_smiles):
    """Generates new LOHC SMILES using the fixed API request format, retrying up to MAX_LLM_ATTEMPTS times if needed."""
    attempt = 0
    unique_smiles = []

    while attempt < MAX_LLM_ATTEMPTS:
        attempt += 1
        print(f"🔹 Attempt {attempt}/{MAX_LLM_ATTEMPTS} to generate new SMILES...")

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
                received_smiles = json_output.get("SMILES", [])
            except Exception:
                received_smiles = extract_smiles(raw_output)

            # Ensure uniqueness
            unique_smiles = list(set(received_smiles))

            if unique_smiles:
                print(f"✅ Received {len(unique_smiles)} unique SMILES after removing duplicates.")
                return extract_smiles(unique_smiles)  # Pass to validation
            else:
                print(f"⚠️ No valid SMILES generated in attempt {attempt}. Retrying...")

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return []

    print("❌ LLM failed to generate valid SMILES after maximum attempts.")
    return []

import time  # Import time module if not already imported

def generate_new_smiles(initial_smiles):
    """Generates new LOHC SMILES using the fixed API request format, retrying up to MAX_LLM_ATTEMPTS times if needed."""
    attempt = 0
    unique_smiles = []

    while attempt < MAX_LLM_ATTEMPTS:
        attempt += 1
        print(f"🔹 Attempt {attempt}/{MAX_LLM_ATTEMPTS} to generate new SMILES...")

        start_time = time.time()  # Start timing

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
                received_smiles = json_output.get("SMILES", [])
            except Exception:
                received_smiles = extract_smiles(raw_output)

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            print(f"⏳ LLM generation time: {elapsed_time:.2f} seconds.")

            # Ensure uniqueness
            unique_smiles = list(set(received_smiles))

            print(f"✅ Received {len(received_smiles)} SMILES from LLM. {len(unique_smiles)} were unique.")

            if unique_smiles:
                return extract_smiles(unique_smiles)  # Pass to validation
            else:
                print(f"⚠️ No valid SMILES generated in attempt {attempt}. Retrying...")

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return []

    print("❌ LLM failed to generate valid SMILES after maximum attempts.")
    return []


def old_filter_and_evaluate_smiles(smiles_list, initial_smiles_set):
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


def old_2_filter_and_evaluate_smiles(smiles_list, initial_smiles_set):
    """Filters generated SMILES based on validity, H2 %, ΔH, and melting point (MP)."""
    valid_smiles = []
    results = []

    print("🔹 Filtering and evaluating new SMILES...")

    for smiles in smiles_list:
        if smiles in initial_smiles_set:
            print(f"❌ {smiles} is in the initial dataset, skipping.")
            continue

        # Check Hydrogen Weight Percentage
        h2_weight = calculate_hydrogen_weight(smiles)
        if h2_weight is None:
            print(f"❌ {smiles} failed H₂ weight requirement.")
            continue

        # Check ΔH using ML model
        delta_h = predict_delta_h(smiles)
        if delta_h is None or not (40 <= delta_h <= 70):
            print(f"❌ {smiles} failed RF ΔH prediction (got {delta_h}).")
            continue

        # Check Melting Point (must be ≤ 40°C or None)
        melting_point = predict_melting_point(smiles)
        if melting_point is not None and melting_point > 40:
            print(f"❌ {smiles} failed melting point filter (MP = {melting_point}°C).")
            continue

        print(f"✅ {smiles} passed all filters. ΔH: {delta_h}, MP: {melting_point if melting_point else 'Unknown'}°C")
        
        # Add valid molecule to results
        valid_smiles.append(smiles)
        results.append({
            "SMILES": smiles,
            "Predicted ΔH": delta_h,
            "Melting_Point": melting_point,
            "pH2": h2_weight
        })

    return valid_smiles, results

def filter_and_evaluate_smiles(smiles_list, initial_smiles_set):
    """Filters generated SMILES based on validity, H2 %, ΔH, and melting point (MP)."""
    valid_smiles = []
    results = []
    total_rejected = 0
    rejection_reasons = {"Invalid structure": 0, "H2 % too low": 0, "ΔH out of range": 0, "High melting point": 0, "Duplicate": 0}

    start_time = time.time()  # Start timing

    print("🔹 Filtering and evaluating new SMILES...")

    for smiles in smiles_list:
        if smiles in initial_smiles_set:
            print(f"❌ {smiles} is in the initial dataset, skipping.")
            rejection_reasons["Duplicate"] += 1
            total_rejected += 1
            continue

        h2_weight = calculate_hydrogen_weight(smiles)
        if h2_weight is None:
            print(f"❌ {smiles} failed H₂ weight requirement.")
            rejection_reasons["H2 % too low"] += 1
            total_rejected += 1
            continue

        delta_h = predict_delta_h(smiles)
        if delta_h is None or not (40 <= delta_h <= 70):
            print(f"❌ {smiles} failed RF ΔH prediction (got {delta_h}).")
            rejection_reasons["ΔH out of range"] += 1
            total_rejected += 1
            continue

        melting_point = predict_melting_point(smiles)
        if melting_point is not None and melting_point > 40:
            print(f"❌ {smiles} failed melting point filter (MP = {melting_point}°C).")
            rejection_reasons["High melting point"] += 1
            total_rejected += 1
            continue

        print(f"✅ {smiles} passed all filters. ΔH: {delta_h}, MP: {melting_point if melting_point else 'Unknown'}°C")
        
        valid_smiles.append(smiles)
        results.append({
            "SMILES": smiles,
            "Predicted ΔH": delta_h,
            "Melting_Point": melting_point,
            "pH2": h2_weight
        })

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"⏳ Filtering & evaluation time: {elapsed_time:.2f} seconds.")

    print(f"🔹 {len(valid_smiles)} molecules passed filtering. {total_rejected} molecules were rejected.")
    print("❌ Rejection breakdown:", rejection_reasons)

    return valid_smiles, results



def old_iterative_generation(initial_set):
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


def old_2_iterative_generation(initial_set):
    """Generates LOHC SMILES iteratively until reaching TARGET_COUNT, ensuring uniqueness at this step."""
    iterative_set = set()
    final_results = []
    final_smiles_set = set()  # Track unique SMILES in results
    initial_smiles_set = set(initial_set)

    for i in range(MAX_ITERATIONS):
        if len(iterative_set) >= TARGET_COUNT:
            break

        print(f"\n🔹 Iteration {i+1}/{MAX_ITERATIONS} - Current set size: {len(iterative_set)}.")

        new_smiles = generate_new_smiles(list(iterative_set) + list(initial_smiles_set))
        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles, initial_smiles_set)

        # Ensure uniqueness in final results
        unique_smiles = [r for r in results if r["SMILES"] not in final_smiles_set]
        
        # Add unique SMILES to tracking sets
        for r in unique_smiles:
            final_smiles_set.add(r["SMILES"])
            iterative_set.add(r["SMILES"])
            final_results.append(r)

        if not unique_smiles:
            print("❌ No unique valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results

def old_3_iterative_generation(initial_set):
    """Generates LOHC SMILES iteratively until reaching TARGET_COUNT, ensuring uniqueness at this step."""
    iterative_set = set()
    final_results = []
    final_smiles_set = set()  # Track unique SMILES across all iterations
    initial_smiles_set = set(initial_set)

    for i in range(MAX_ITERATIONS):
        if len(iterative_set) >= TARGET_COUNT:
            break

        print(f"\n🔹 Iteration {i+1}/{MAX_ITERATIONS} - Current set size: {len(iterative_set)}.")

        new_smiles = generate_new_smiles(list(iterative_set) + list(initial_smiles_set))
        
        # Ensure canonical form before filtering
        new_smiles = [is_valid_smiles(s) for s in new_smiles if s]  # Convert to canonical SMILES
        new_smiles = list(set(new_smiles))  # Remove duplicates immediately

        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles, initial_smiles_set)

        # Ensure uniqueness in final results
        unique_smiles = [r for r in results if r["SMILES"] not in final_smiles_set]

        # Add unique SMILES to tracking sets
        for r in unique_smiles:
            final_smiles_set.add(r["SMILES"])
            iterative_set.add(r["SMILES"])
            final_results.append(r)

        if not unique_smiles:
            print("❌ No unique valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results


def old_4_iterative_generation(initial_set):
    """Generates LOHC SMILES iteratively until reaching TARGET_COUNT, ensuring uniqueness at this step."""
    iterative_set = set()
    final_results = []
    final_smiles_set = set()  # Track unique SMILES across all iterations
    initial_smiles_set = set(initial_set)

    for i in range(MAX_ITERATIONS):
        if len(iterative_set) >= TARGET_COUNT:
            break

        print(f"\n🔹 Iteration {i+1}/{MAX_ITERATIONS} - Current set size: {len(iterative_set)}.")

        new_smiles = generate_new_smiles(list(iterative_set) + list(initial_smiles_set))
        
        # Ensure canonical form before filtering
        new_smiles = [is_valid_smiles(s) for s in new_smiles if s]  # Convert to canonical SMILES
        new_smiles = list(set(new_smiles))  # Remove duplicates immediately

        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles, initial_smiles_set)

        # Ensure uniqueness in final results
        unique_smiles = [r for r in results if r["SMILES"] not in final_smiles_set]

        # Add additional properties (pH2 and nH2)
        for r in unique_smiles:
            smiles = r["SMILES"]
            h2_weight = calculate_hydrogen_weight(smiles)
            
            # Calculate MW values
            mw_lean = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
            hydrogenated_smiles = hydrogenate_smiles(smiles)
            mw_rich = Descriptors.MolWt(Chem.MolFromSmiles(hydrogenated_smiles))

            nH2 = (mw_rich - mw_lean) / 2  # Compute number of H2 molecules
            
            # Add to results
            r["pH2"] = h2_weight  # Hydrogen weight percentage
            r["nH2"] = nH2  # Number of hydrogen molecules

            final_smiles_set.add(smiles)
            iterative_set.add(smiles)
            final_results.append(r)

        if not unique_smiles:
            print("❌ No unique valid SMILES generated. Consider adjusting parameters.")
            break

    return final_results

def iterative_generation(initial_set):
    """Generates LOHC SMILES iteratively until reaching TARGET_COUNT, ensuring uniqueness at this step."""
    iterative_set = set()
    final_results = []
    final_smiles_set = set()
    initial_smiles_set = set(initial_set)

    for i in range(MAX_ITERATIONS):
        if len(iterative_set) >= TARGET_COUNT:
            break

        start_time = time.time()  # Start iteration timer
        print(f"\n🔹 Iteration {i+1}/{MAX_ITERATIONS} - Current set size: {len(iterative_set)}.")

        new_smiles = generate_new_smiles(list(iterative_set) + list(initial_smiles_set))
        
        new_smiles = [is_valid_smiles(s) for s in new_smiles if s]  
        new_smiles = list(set(new_smiles))  

        filtered_smiles, results = filter_and_evaluate_smiles(new_smiles, initial_smiles_set)

        unique_smiles = [r for r in results if r["SMILES"] not in final_smiles_set]

        for r in unique_smiles:
            final_smiles_set.add(r["SMILES"])
            iterative_set.add(r["SMILES"])
            final_results.append(r)

        end_time = time.time()  # End iteration timer
        elapsed_time = end_time - start_time
        print(f"⏳ Iteration {i+1} completed in {elapsed_time:.2f} seconds.")

        if not unique_smiles:
            print("❌ No unique valid SMILES generated. Consider adjusting parameters.")
            break

    print(f"✅ Final dataset contains {len(final_results)} unique LOHC molecules.")
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

