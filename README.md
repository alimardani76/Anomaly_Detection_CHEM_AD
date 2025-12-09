# CHEM-AD Anomaly Detection in Metal-Organic Frameworks (MOFs)

> This repository contains a comprehensive methodology for anomaly detection in MOF structures using autoencoders. The project includes a full hyperparameter search, feature justification, model validation, and visualization of results.

---

## Dataset

This project is based on two primary data files:

* **`dataset.csv`**: The primary dataset. It contains 32 numeric descriptors (chemical, topological, and geometrical) for each MOF, plus 49 multi-hot encoded features indicating the presence of different metals.
* **`df_anomalies.csv`**: A file containing pre-calculated anomaly scores. It provides scores for each individual descriptor as well as an average total anomaly score per MOF.

### Data Source

The chemical, topological, and geometrical descriptors used in `dataset.csv` were originally sourced from the **mofxdb** database.

[https://mof.tech.northwestern.edu/](https://mof.tech.northwestern.edu/)

---

## Methodology

The core of this project is a deep learning approach to identify anomalies. The methodology is broken into three parts: the main model selection, a feature justification analysis, and a model consistency check.

### 1. Experiment Model Selection

This directory holds the comprehensive output from the autoencoder hyperparameter search. The script performs a grid search by iterating through different `latent_dim` (latent dimension sizes) and `step_size` (which defines the encoder/decoder layer structure). The model is a standard `Dense` autoencoder using `BatchNormalization` and `ReLU` activations.

The experiment is run using a 5-fold split setup. Each numbered file (e.g., `results_0.csv`, `results_1.csv`) contains the raw results for a single fold, while the `analysis_results_...csv` files provide aggregated and flattened summaries.

Key metrics captured for each configuration include:

* **`val_loss`**: Final validation loss from training.
* **`test_error`**: Mean squared error on the test set.
* **`anomaly_threshold`**: The threshold calculated using the **elbow method** on the sorted reconstruction errors of the training set.
* **`maha_dist_ratio_95`**: Mahalanobis distance ratio (anomaly vs. normal) calculated in the 95% variance PCA space.
* **`anomaly_percent`**: The percentage of test samples flagged as anomalies by the calculated threshold.

### 2. Jaccard Similarity Analysis (Feature Justification)

This notebook provides a supporting analysis to justify the model's design. It explores whether anomalies can be effectively separated *without* using the 49-feature metal composition set.

The analysis uses **Jaccard similarity** and **containment** metrics on the non-metal features. The consistent results suggest that the chemical, topological, and geometrical descriptors alone are highly effective at distinguishing anomalies, which validates the main model's findings.

### 3. Anomaly Ranking & Jaccard Comparison (Model Consistency)

This analysis, located in the `Experiment Model Selection/Ranking of Top Anomalies` folder, compares the consistency of anomaly detection across all the different autoencoder models (both shallow and deep) from the main experiment.

Using the reconstruction errors and anomaly thresholds for each model, this script identifies the **top 100 anomalies** predicted by each configuration. It then calculates the **Jaccard index** to measure the overlap between every pair of models. The results are saved in **`jaccard.csv`**, which contains the following columns:

* **`model_A`**, **`model_B`**: The pair of models being compared.
* **`mdr_A`**, **`mdr_B`**: The Mahalanobis distance ratio (a performance-related metric) for each model.
* **`jaccard_index`**: The Jaccard similarity score for their top-100 anomaly sets.
* **`overlap_count`**: The raw number of anomalies common to both models.

The key finding is that despite varying depths and architectures, all models show significant overlap, confirming a consistent set of core anomalies.

---

## Results & Visualizations

The `Dimension Reduction` ,  folders contains the visualizations including various dimension reduction techniques, including **PCA**, **t-SNE**, and **UMAP**. All plots are colored by MOF category (anomaly vs. non-anomaly) to visualize cluster separation. Additionally, the PCA plots include versions that specifically mark and highlight the top 10 most anomalous MOFs identified by the models.
The `Results` , `Figures` contain pairwise correlation heatmap, feature anomaly contribution, feature distribution (anomaly vs normal MoFs).



# 🔬 MOF Feature Extractor for Machine Learning 🔬

This project provides a set of Python scripts to extract a comprehensive set of geometric, chemical, and topological features from Metal-Organic Framework (MOF) structural files (`.cif` and `.json`). The resulting feature sets are saved as `.csv` files, ready for use in machine learning models to predict MOF properties.

The pipeline is designed to work with large datasets like the **CoRE MOF 2019** and **hMOF** databases.

## 🔬 Features Extracted

This pipeline extracts over 50 features, categorized as follows:

* **Geometric:** Surface area, void fraction, pore limiting diameter (PLD), and largest cavity diameter (LCD).
* **Chemical:** Density, formula, elemental properties (e.g., average electronegativity), metal fractions, and one-hot encoding of common metals.
* **Topological:** Graph-based properties describing the MOF's connectivity, such as graph density, diameter, average shortest path length, and clustering coefficients.
* **Linker & Metal:** Properties specific to the organic linkers and metal centers, such as average bond lengths and metal coordination numbers.

## 🚀 Workflow & How to Use

The data processing is broken down into a sequence of scripts. You should run them in order, as each script's output may be the input for the next one.

### **Prerequisites**

Make sure you have the required Python libraries installed:

```bash
pip install pandas pymatgen networkx tqdm
```

You will also need to have your raw MOF dataset folders (e.g., `CoREMOF 2019`, `hMOF-10_CO2_CH4_N2`) in the same directory as these scripts.

### **Step-by-Step Instructions(Feature_extraction Folder)**

1.  **Prepare the Dataset (`01_prepare_dataset.py`)**
    * **What it does:** Finds all `.cif` files in your raw dataset folders, matches them with their corresponding `.json` files, and copies them into a clean project directory (`MOFxDB_Project`).
    * **How to run:**
        ```bash
        python 01_prepare_dataset.py
        ```

2.  **Extract Geometric Features (`02_extract_geometric_features.py`)**
    * **What it does:** Reads the `.json` files to extract pre-calculated geometric properties.
    * **How to run:**
        ```bash
        python 02_extract_geometric_features.py
        ```

3.  **Extract Chemical Features (`03_extract_chemical_features.py`)**
    * **What it does:** Analyzes the `.cif` files to calculate a wide range of chemical and compositional properties. This script processes the full CoRE MOF set and a 15k subset of hMOFs.
    * **How to run:**
        ```bash
        python 03_extract_chemical_features.py
        ```

4.  **Extract Topological Features (`04_extract_topological_features.py`)**
    * **What it does:** Treats each MOF as a mathematical graph to calculate features describing its connectivity and topology.
    * **How to run:**
        ```bash
        python 04_extract_topological_features.py
        ```

5.  **Extract Linker & Metal Features (`05_extract_linker_metal_features.py`)**
    * **What it does:** Calculates features specific to the organic linkers and metal nodes within the MOF structures.
    * **How to run:**
        ```bash
        python 05_extract_linker_metal_features.py
        ```

After running all the scripts, your extracted features will be located in the `MOFxDB_Project/features/` directory, ready for your analysis!
