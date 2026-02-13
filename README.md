<img width="1645" height="1536" alt="Gemini_Generated_Image_xbqbmfxbqbmfxbqb" src="https://github.com/user-attachments/assets/b9d7c932-c730-4c44-943d-6f848c926ec5" />

# 🔬 CHEM-AD: Deep Learning-Driven Anomaly Detection for Metal-Organic Frameworks

**CHEM-AD** is a comprehensive computational framework for identifying structural and chemical anomalies in Metal-Organic Framework (MOF) databases. By utilizing deep autoencoders, the project provides a systematic way to detect "outlier" materials—whether they are physical anomalies (unstable/erroneous structures) or chemical anomalies (materials with unique, high-value properties).

---

## 📂 Core Dataset Analysis

The framework operates on high-dimensional data derived from the **mofxdb** database, covering chemical, topological, and geometrical descriptors.

* **`dataset.csv`**: The primary input file. It comprises 32 numeric descriptors and 49 multi-hot encoded metal features.
* **`df_anomalies.csv`**: The output repository containing calculated anomaly scores for every descriptor and an aggregated total anomaly score for each MOF.

---

## 🧠 Methodology & Experimental Design

The detection pipeline is validated through a three-stage experimental process to ensure that the identified anomalies are statistically significant and model-independent.

### 1. Experiment Model Selection (Deep Autoencoders)

This stage involves a rigorous hyperparameter grid search to identify the optimal **Dense Autoencoder** architecture.

* **Architecture:** The models use `BatchNormalization` and `ReLU` activations to learn a compressed representation of the MOF feature space.
* **Grid Search:** Iterates through various `latent_dim` (bottleneck size) and `step_size` (layer scaling) configurations.
* **Thresholding:** Anomalies are determined using the **elbow method** on sorted reconstruction errors.
* **Validation:** Performance is measured via `val_loss`, `test_error`, and the **Mahalanobis Distance Ratio (`maha_dist_ratio_95`)** in a 95% variance PCA space.

### 2. Jaccard Similarity Analysis (Feature Justification)

To ensure the model isn't biased by metal composition, this analysis tests if anomalies can be detected using *only* geometric and topological descriptors. High **Jaccard similarity** and **containment** scores between these models justify the exclusion of metal features when focusing purely on structural "oddities."

### 3. Anomaly Ranking & Consistency

Located in `Experiment Model Selection/Ranking of Top Anomalies`, this script assesses the **top 100 anomalies** across all model architectures. By generating a **`jaccard.csv`** matrix, we demonstrate that despite variations in model depth, the "core" anomalies are consistently flagged, proving the methodology's reliability.

---

## 📊 Results & Visual Analytics

The framework generates several high-fidelity visualizations to aid in the interpretation of results:

* **`Dimension Reduction`**: Contains **PCA**, **t-SNE**, and **UMAP** plots. These visualize how anomalies separate from the "normal" population in lower-dimensional space.
* **`Figures` & `Results**`:
* **Heatmaps**: Pairwise correlation of features.
* **Contribution Plots**: Breakdown of which specific features (e.g., pore size, density) contributed most to a MOF being flagged as an anomaly.
* **Distributions**: Comparative histograms of feature ranges between normal and anomalous MOFs.



---

## 🛠 Feature Extraction Pipeline

This sub-module (located in the `Feature_extraction` folder) provides a sequence of Python scripts to transform raw `.cif` or `.json` structural files into ML-ready datasets.

### **Step-by-Step Execution**

1. **`01_prepare_dataset.py`**: Organizes raw CoRE MOF and hMOF files into a unified project directory (`MOFxDB_Project`).
2. **`02_extract_geometric_features.py`**: Extracts pre-calculated geometric data (Surface Area, Void Fraction, etc.) from metadata files.
3. **`03_extract_chemical_features.py`**: Uses `pymatgen` to analyze elemental compositions and density.
4. **`04_extract_topological_features.py`**: Converts structures into graphs to calculate connectivity metrics like clustering coefficients and graph diameter.
5. **`05_extract_linker_metal_features.py`**: Isolates and describes the specific chemistry of organic linkers and metal nodes.

---

## 📝 Preprint & Citation

The theoretical background, detailed feature importance analysis, and chemical justification for these findings can be found in our preprint:

**Title:** *Decoding the Unseen: Unsupervised Anomaly Detection in Metal–Organic Frameworks for Discovery Beyond the Norm* **Preprint Link:** [https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-hhr97-v3](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-hhr97-v3)

```text
Please cite as:
[Alimardani et al.], "Decoding the Unseen: Unsupervised Anomaly Detection in Metal–Organic Frameworks for Discovery Beyond the Norm", 
ChemRxiv (2025). DOI: 10.26434/chemrxiv-2025-hhr97-v3

```

---

**Maintainer:** [Hosein Alimardani mailto:Hosein.alimardani76@gmail.com]
