import re
import pandas as pd

# Reload the file
log_path_expt = "output_expt.log"

# Define lists to store extracted data
iterations_expt = []
gen_times_expt = []
filter_times_expt = []
total_times_expt = []
num_received_expt = []
num_unique_expt = []
num_passed_expt = []
num_rejected_expt = []
rejection_breakdowns_expt = []

# Read log file
with open(log_path_expt, 'r', encoding='utf-8') as file:
    lines_expt = file.readlines()

# Regex patterns for extraction
iteration_pattern = re.compile(r"🔹 Iteration (\d+)/\d+")
gen_time_pattern = re.compile(r"LLM generation time: ([\d.]+) seconds")
filter_time_pattern = re.compile(r"Filtering & evaluation time: ([\d.]+) seconds")
total_time_pattern = re.compile(r"Iteration \d+ completed in ([\d.]+) seconds")
received_pattern = re.compile(r"Received (\d+) SMILES from LLM")
unique_pattern = re.compile(r"(\d+) were unique")
passed_pattern = re.compile(r"(\d+) molecules passed filtering")
rejected_pattern = re.compile(r"(\d+) molecules were rejected")
rejection_breakdown_pattern = re.compile(r"Rejection breakdown: (.*)")

# Iterate over lines to extract data
for line in lines_expt:
    iteration_match = iteration_pattern.search(line)
    if iteration_match:
        iterations_expt.append(int(iteration_match.group(1)))

    gen_time_match = gen_time_pattern.search(line)
    if gen_time_match:
        gen_times_expt.append(float(gen_time_match.group(1)))

    filter_time_match = filter_time_pattern.search(line)
    if filter_time_match:
        filter_times_expt.append(float(filter_time_match.group(1)))

    total_time_match = total_time_pattern.search(line)
    if total_time_match:
        total_times_expt.append(float(total_time_match.group(1)))

    received_match = received_pattern.search(line)
    if received_match:
        num_received_expt.append(int(received_match.group(1)))

    unique_match = unique_pattern.search(line)
    if unique_match:
        num_unique_expt.append(int(unique_match.group(1)))

    passed_match = passed_pattern.search(line)
    if passed_match:
        num_passed_expt.append(int(passed_match.group(1)))

    rejected_match = rejected_pattern.search(line)
    if rejected_match:
        num_rejected_expt.append(int(rejected_match.group(1)))

    rejection_match = rejection_breakdown_pattern.search(line)
    if rejection_match:
        rejection_breakdowns_expt.append(rejection_match.group(1))

# Ensure all lists have the same length
max_length_expt = max(len(iterations_expt), len(gen_times_expt), len(filter_times_expt), len(total_times_expt),
                       len(num_received_expt), len(num_unique_expt), len(num_passed_expt), len(num_rejected_expt), len(rejection_breakdowns_expt))

# Function to pad lists
def pad_list(lst, length):
    return lst + [None] * (length - len(lst))

# Pad lists
iterations_expt = pad_list(iterations_expt, max_length_expt)
gen_times_expt = pad_list(gen_times_expt, max_length_expt)
filter_times_expt = pad_list(filter_times_expt, max_length_expt)
total_times_expt = pad_list(total_times_expt, max_length_expt)
num_received_expt = pad_list(num_received_expt, max_length_expt)
num_unique_expt = pad_list(num_unique_expt, max_length_expt)
num_passed_expt = pad_list(num_passed_expt, max_length_expt)
num_rejected_expt = pad_list(num_rejected_expt, max_length_expt)
rejection_breakdowns_expt = pad_list(rejection_breakdowns_expt, max_length_expt)

# Create DataFrame
df_expt = pd.DataFrame({
    "Iteration": iterations_expt,
    "LLM Generation Time (s)": gen_times_expt,
    "Filtering & Evaluation Time (s)": filter_times_expt,
    "Total Iteration Time (s)": total_times_expt,
    "SMILES Received": num_received_expt,
    "SMILES Unique": num_unique_expt,
    "Molecules Passed": num_passed_expt,
    "Molecules Rejected": num_rejected_expt
})

# Extract individual rejection categories
rejection_categories = ["Invalid structure", "H2 % too low", "ΔH out of range", "High melting point", "Duplicate"]
for category in rejection_categories:
    df_expt[category] = 0

# Populate the columns with actual values from the rejection breakdown
for index, row in df_expt.iterrows():
    breakdown = eval(rejection_breakdowns_expt[index]) if rejection_breakdowns_expt[index] else {}
    for category in rejection_categories:
        df_expt.at[index, category] = breakdown.get(category, 0)

# Compute SMILES Unique (Recalculated) as sum of Molecules Passed and Molecules Rejected
df_expt["SMILES Unique (Recalculated)"] = df_expt["Molecules Passed"] + df_expt["Molecules Rejected"]

# Compute Total Rejected (Calculated) as sum of individual rejection categories
df_expt["Total Rejected (Calculated)"] = (
    df_expt["Invalid structure"] + df_expt["H2 % too low"] + df_expt["ΔH out of range"] +
    df_expt["High melting point"] + df_expt["Duplicate"]
)

# Add a verification column to check if the calculated total matches Molecules Rejected
df_expt["Matches Molecules Rejected"] = df_expt["Total Rejected (Calculated)"] == df_expt["Molecules Rejected"]
df_expt["Matches Molecules Rejected"] = df_expt["Matches Molecules Rejected"].map({True: "Yes", False: "No"})

# Save DataFrame to CSV
csv_path = "parsed_output_expt.csv"
df_expt.to_csv(csv_path, index=False)

# Display the DataFrame for verification

