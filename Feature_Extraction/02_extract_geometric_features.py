# 02_extract_geometric_features.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_geometric_features():
    """
    Reads all .json files from the organized project folder and extracts
    key geometric features like surface area, void fraction, and pore diameters.
    Saves the features to a CSV file and plots a summary of any missing data.
    """
    print("\n--- Step 2: Extracting Geometric Features ---")

    # --- CONFIGURATION ---
    json_folder = os.path.join("MOFxDB_Project", "jsons")
    output_dir = os.path.join("MOFxDB_Project", "features", "geometric")
    output_csv = os.path.join(output_dir, "geometric_features.csv")

    # Define the specific geometric keys to extract from the JSON files
    feature_keys = [
        "surface_area_m2g",
        "surface_area_m2cm3",
        "void_fraction",
        "pld",
        "lcd",
    ]

    # --- PREPARATION ---
    if not os.path.isdir(json_folder):
        print(f"❌ Error: The JSON folder was not found at '{json_folder}'.")
        print("   Please run '01_prepare_dataset.py' first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = []
    missing_counts = defaultdict(int)

    # --- PROCESS JSON FILES ---
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"🔍 Found {len(json_files)} JSON files to process.")

    for filename in json_files:
        json_path = os.path.join(json_folder, filename)
        mof_id = filename.replace(".json", "")
        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)

            row = {"MOF_ID": mof_id}
            for key in feature_keys:
                value = json_data.get(key)
                if value is None:
                    missing_counts[key] += 1
                row[key] = value
            data.append(row)

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    # --- SAVE CSV ---
    if not data:
        print("❌ Error: No data was extracted. Check the JSON files for content.")
        return

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Geometric features saved to: '{output_csv}'")

    # --- PLOT MISSING DATA ---
    if missing_counts:
        print("\n📊 Plotting summary of missing features...")
        plt.figure(figsize=(10, 5))
        plt.bar(missing_counts.keys(), missing_counts.values(), color='skyblue')
        plt.ylabel("Number of Missing Values")
        plt.title("Missing Geometric Features Across All MOFs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_geometric_features.png"))
        plt.show()
    else:
        print("[✓] No missing geometric features were detected.")

if __name__ == "__main__":
    extract_geometric_features()