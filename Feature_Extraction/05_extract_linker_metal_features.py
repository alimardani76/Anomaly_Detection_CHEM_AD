# 05_extract_linker_metal_features.py
import os
import pandas as pd
import numpy as np
import multiprocessing
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_metal(symbol):
    """Checks if an element symbol corresponds to a metal."""
    try:
        el = Element(symbol)
        return el.is_metal
    except:
        return False

def extract_linker_metal_features(cif_path_tuple):
    """
    Processes a single CIF file to extract features related to the metal centers
    and the organic linkers connecting them.
    """
    cif_folder, cif_file, mof_id = cif_path_tuple
    cif_path = os.path.join(cif_folder, cif_file)

    try:
        struct = Structure.from_file(cif_path)
        cnn = CrystalNN()

        # Identify metal and linker atoms
        metal_sites_indices = [i for i, site in enumerate(struct) if is_metal(site.specie.symbol)]
        
        # Calculate linker atom fraction
        linker_frac = 1.0 - (len(metal_sites_indices) / len(struct)) if len(struct) > 0 else np.nan

        # Calculate metal coordination numbers
        metal_coord_numbers = []
        for idx in metal_sites_indices:
            try:
                env = cnn.get_nn_info(struct, idx)
                metal_coord_numbers.append(len(env))
            except Exception:
                continue # Skip if coordination env can't be found

        return {
            "MOF_ID": mof_id,
            "linker_atom_fraction": linker_frac,
            "metal_coord_number_mean": np.mean(metal_coord_numbers) if metal_coord_numbers else np.nan,
            "metal_coord_number_std": np.std(metal_coord_numbers) if metal_coord_numbers else np.nan,
        }

    except Exception:
        return {
            "MOF_ID": mof_id,
            "linker_atom_fraction": np.nan,
            "metal_coord_number_mean": np.nan,
            "metal_coord_number_std": np.nan,
        }

def run_linker_metal_extraction():
    """
    Main function to orchestrate the parallel extraction of linker and metal features,
    processing only the MOFs that were successful in the previous step.
    """
    print("\n--- Step 5: Extracting Linker & Metal Features ---")

    # --- CONFIGURATION ---
    cif_folder = os.path.join("MOFxDB_Project", "cifs")
    subset_csv = os.path.join("MOFxDB_Project", "features", "chemical", "chemical_features.csv")
    output_dir = os.path.join("MOFxDB_Project", "features", "linker_metal")
    output_csv = os.path.join(output_dir, "linker_metal_features.csv")
    N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    # --- PREPARATION ---
    if not os.path.exists(subset_csv):
        print(f"❌ Error: The chemical features file was not found at '{subset_csv}'.")
        print("   Please run '03_extract_chemical_features.py' first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load MOF IDs from the previous step
    subset_df = pd.read_csv(subset_csv)
    mof_ids_to_process = set(subset_df["MOF_ID"].dropna().astype(str))
    
    tasks = [
        (cif_folder, f"{mof_id}.cif", mof_id)
        for mof_id in mof_ids_to_process
        if os.path.exists(os.path.join(cif_folder, f"{mof_id}.cif"))
    ]
    
    print(f"🧪 Processing {len(tasks)} CIF files with {N_WORKERS} workers...")

    # --- PARALLEL EXECUTION ---
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(extract_linker_metal_features, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Linker/Metal Features"):
            result = future.result()
            if result:
                results.append(result)

    # --- SAVE RESULTS ---
    if not results:
        print("\n❌ Error: No linker/metal features were extracted.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Linker and metal features saved to: '{output_csv}'")

if __name__ == "__main__":
    run_linker_metal_extraction()