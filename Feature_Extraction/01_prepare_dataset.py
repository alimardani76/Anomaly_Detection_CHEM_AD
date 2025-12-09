# 01_prepare_dataset.py
import os
import pandas as pd
import shutil

def prepare_dataset():
    """
    Finds all .cif and .json files in the source dataset folders,
    matches them, and copies them into a centralized project directory.
    It also creates a master CSV file that tracks all structures.
    """
    print("--- Step 1: Preparing Dataset ---")

    # --- CONFIGURATION ---
    # Define the source folders for your raw MOF data
    input_folders = {
        "core2019": "CoREMOF 2019",
        "hmof_co2_ch4_n2": "hMOF-10_CO2_CH4_N2",
        "hmof_h2": "hMOF-10_H2",
        "hmof_kr_xe": "hMOF-10_Kr_Xe_Selectivity"
    }

    # Define the target folders for the organized project
    project_dir = "MOFxDB_Project"
    cif_out_dir = os.path.join(project_dir, "cifs")
    json_out_dir = os.path.join(project_dir, "jsons")
    output_csv = os.path.join(project_dir, "mof_master_list.csv")

    # Create directories if they don't exist
    os.makedirs(cif_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)

    # --- FILE DISCOVERY AND COPYING ---
    records = []
    for label, folder in input_folders.items():
        if not os.path.isdir(folder):
            print(f"⚠️ Warning: Source folder not found: '{folder}'. Skipping.")
            continue

        print(f"🔍 Searching in: {folder}...")
        for file in os.listdir(folder):
            if file.endswith(".cif"):
                mof_id = os.path.splitext(file)[0]
                json_name = f"{mof_id}.json"
                cif_path = os.path.join(folder, file)
                json_path = os.path.join(folder, json_name)

                # Ensure a matching JSON file exists before adding
                if os.path.exists(json_path):
                    # Copy files to the centralized project folders
                    shutil.copy2(cif_path, os.path.join(cif_out_dir, file))
                    shutil.copy2(json_path, os.path.join(json_out_dir, json_name))

                    # Add a record for the master CSV file
                    records.append({
                        "MOF_ID": mof_id,
                        "dataset_origin": label,
                        "path_to_cif": os.path.join("cifs", file),
                        "path_to_json": os.path.join("jsons", json_name)
                    })
                else:
                    print(f"   - Note: JSON file missing for {mof_id}, skipping.")

    # --- SAVE MASTER LIST ---
    if not records:
        print("\n❌ Error: No matching .cif/.json pairs were found. Please check your `input_folders` paths.")
        return

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Finished! Found and organized {len(records)} MOFs.")
    print(f"   - CIFs and JSONs copied to '{project_dir}/'")
    print(f"   - Master list saved to: '{output_csv}'")

if __name__ == "__main__":
    prepare_dataset()