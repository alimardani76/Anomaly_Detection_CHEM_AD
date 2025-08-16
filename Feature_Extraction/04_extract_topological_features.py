# 04_extract_topological_features.py
import os
import pandas as pd
import numpy as np
import multiprocessing
import logging
import networkx as nx
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

def extract_topo_features(cif_path_tuple):
    """
    Processes a single CIF file to extract topological features by representing
    the MOF structure as a mathematical graph.
    """
    cif_folder, cif_file, mof_id = cif_path_tuple
    cif_path = os.path.join(cif_folder, cif_file)

    try:
        # Build the graph representation of the MOF
        struct = Structure.from_file(cif_path)
        cnn = CrystalNN()
        sg = StructureGraph.with_local_env_strategy(struct, cnn)
        g = nx.Graph(sg.graph)

        is_connected = nx.is_connected(g)

        # Calculate degrees and components
        degrees = [d for _, d in g.degree()]
        components = list(nx.connected_components(g))
        largest_cc = max(components, key=len) if components else []
        largest_cc_fraction = len(largest_cc) / g.number_of_nodes() if g.number_of_nodes() > 0 else 0

        # Graph entropy calculation
        degree_counts = Counter(dict(g.degree()).values())
        total_nodes = sum(degree_counts.values())
        probs = np.array([count / total_nodes for count in degree_counts.values()])
        entropy = -np.sum(probs * np.log2(probs, where=probs>0))

        # Assemble feature dictionary
        result = {
            "MOF_ID": mof_id,
            "avg_node_connectivity": np.mean(degrees) if degrees else 0,
            "graph_density": nx.density(g),
            "num_connected_components": nx.number_connected_components(g),
            "largest_cc_fraction": largest_cc_fraction,
            "is_connected": int(is_connected),
            "clustering_coefficient_mean": np.mean(list(nx.clustering(g).values())),
            "degree_assortativity": nx.degree_assortativity_coefficient(g),
            "graph_transitivity": nx.transitivity(g),
            "graph_entropy": entropy,
        }

        # Features that only work for fully connected graphs
        if is_connected:
            result["graph_diameter"] = nx.diameter(g)
            result["graph_radius"] = nx.radius(g)
            result["avg_shortest_path_length"] = nx.average_shortest_path_length(g)
        else:
            result["graph_diameter"] = -1
            result["graph_radius"] = -1
            result["avg_shortest_path_length"] = -1

        return result

    except Exception as e:
        logging.error(f"Error processing {mof_id}: {e}")
        return {"MOF_ID": mof_id} # Return at least the ID for merging

def extract_topological_features():
    """
    Main function to orchestrate the parallel extraction of topological features.
    It only processes the MOFs that were successfully processed in the chemical step.
    """
    print("\n--- Step 4: Extracting Topological Features ---")

    # --- CONFIGURATION ---
    cif_folder = os.path.join("MOFxDB_Project", "cifs")
    subset_csv = os.path.join("MOFxDB_Project", "features", "chemical", "chemical_features.csv")
    output_dir = os.path.join("MOFxDB_Project", "features", "topological")
    output_csv = os.path.join(output_dir, "topological_features.csv")
    error_log = os.path.join(output_dir, "topological_extraction_errors.log")
    N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    # --- PREPARATION ---
    if not os.path.exists(subset_csv):
        print(f"❌ Error: The chemical features file was not found at '{subset_csv}'.")
        print("   Please run '03_extract_chemical_features.py' first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=error_log, level=logging.ERROR, format='%(message)s')

    # Load MOF IDs from the previous step to ensure we only process valid structures
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
        futures = {executor.submit(extract_topo_features, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Topology"):
            result = future.result()
            if result:
                results.append(result)

    # --- SAVE RESULTS ---
    if not results:
        print("\n❌ Error: No topological features were extracted.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Extracted features from {len(df[df.columns.drop('MOF_ID')].dropna())} MOFs.")
    print(f"   - Output saved to: '{output_csv}'")
    if os.path.getsize(error_log) > 0:
        print(f"   - Some errors occurred. See log: '{error_log}'")

if __name__ == "__main__":
    extract_topological_features()