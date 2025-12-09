# 03_extract_chemical_features.py
import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Suppress Pymatgen warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def is_metal(symbol):
    """Checks if an element symbol corresponds to a metal commonly found in MOFs."""
    try:
        element = Element(symbol)
        return element.is_transition_metal or element.is_alkaline or element.is_alkali or element.is_post_transition_metal or element.is_lanthanoid
    except:
        return False

def process_cif(cif_path_tuple):
    """
    Processes a single CIF file to extract a wide range of chemical features.
    Designed to be run in parallel.
    """
    cif_folder, filename = cif_path_tuple
    mof_id = filename.replace(".cif", "")
    cif_path = os.path.join(cif_folder, filename)

    try:
        # Load structure and composition
        structure = Structure.from_file(cif_path)
        comp = structure.composition
        el_amt = dict(comp.get_el_amt_dict())
        total_atoms = sum(el_amt.values())

        # Calculate features
        volume = structure.volume
        density = structure.density

        # Electronegativity features
        en_values = [Element(el).X for el, w in el_amt.items() if Element(el).X is not None]
        en_weights = [w for el, w in el_amt.items() if Element(el).X is not None]
        avg_en = np.average(en_values, weights=en_weights) if en_values else np.nan
        var_en = np.average([(x - avg_en) ** 2 for x in en_values], weights=en_weights) if en_values else np.nan

        # Metal features
        metal_atoms = sum(amt for el, amt in el_amt.items() if is_metal(el))
        metal_fraction = metal_atoms / total_atoms if total_atoms > 0 else np.nan

        # Space group
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group, sg_num = sga.get_space_group_info()
        except:
            space_group, sg_num = None, None

        # Assemble feature dictionary
        features = {
            "MOF_ID": mof_id,
            "formula": comp.reduced_formula,
            "num_atoms": total_atoms,
            "volume": volume,
            "density": density,
            "avg_electronegativity": avg_en,
            "electronegativity_variance": var_en,
            "metal_fraction": metal_fraction,
            "num_unique_elements": len(el_amt),
            "metal_atom_count": metal_atoms,
            "space_group": space_group,
            "space_group_number": sg_num
        }

        # One-hot encode common metals
        common_metals = ["Zn", "Cu", "Zr", "Fe", "Co", "Ni", "Mn", "Cr", "V", "Al", "Mg", "Ca"]
        for metal in common_metals:
            features[f"metal_{metal}"] = 1 if metal in el_amt else 0

        return features, None

    except Exception as e:
        return None, (filename, str(e))

def extract_chemical_features():
    """
    Main function to orchestrate the parallel extraction of chemical features
    from all CoRE MOFs and a 15k subset of hMOFs.
    """
    print("\n--- Step 3: Extracting Chemical Features ---")

    # --- CONFIGURATION ---
    cif_folder = os.path.join("MOFxDB_Project", "cifs")
    output_dir = os.path.join("MOFxDB_Project", "features", "chemical")
    output_csv = os.path.join(output_dir, "chemical_features.csv")
    N_WORKERS = max(1, os.cpu_count() - 1)

    # --- PREPARATION ---
    if not os.path.isdir(cif_folder):
        print(f"❌ Error: The CIF folder was not found at '{cif_folder}'.")
        print("   Please run '01_prepare_dataset.py' first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Select all CoRE MOFs and a 15k subset of hMOFs
    all_files = os.listdir(cif_folder)
    core_files = [f for f in all_files if f.endswith(".cif") and not f.startswith("hMOF-")]
    hmof_files = [f for f in all_files if f.endswith(".cif") and f.startswith("hMOF-")][:15000]
    cif_files_to_process = core_files + hmof_files
    
    print(f"🧪 Processing {len(cif_files_to_process)} CIF files with {N_WORKERS} workers...")

    # --- PARALLEL EXECUTION ---
    results = []
    errors = []
    tasks = [(cif_folder, fname) for fname in cif_files_to_process]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_cif, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIFs"):
            result, error = future.result()
            if result:
                results.append(result)
            if error:
                errors.append(error)

    # --- SAVE RESULTS ---
    if not results:
        print("\n❌ Error: No chemical features were extracted. Check for errors during processing.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Extracted features from {len(results)} / {len(cif_files_to_process)} CIFs.")
    print(f"   - Output saved to: '{output_csv}'")
    if errors:
        print(f"   - Skipped {len(errors)} files due to errors. See error log for details.")
        with open(os.path.join(output_dir, "chemical_extraction_errors.log"), "w") as f:
            for fname, reason in errors:
                f.write(f"{fname}: {reason}\n")

if __name__ == "__main__":
    extract_chemical_features()