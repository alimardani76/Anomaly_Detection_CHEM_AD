# Anomaly Detection in Metal-Organic Frameworks (MOFs)

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

The `Dimension Reduction Visualizations` folder contains the final visualization plots from various dimension reduction techniques, including **PCA**, **t-SNE**, and **UMAP**.

All plots are colored by MOF category (anomaly vs. non-anomaly) to visualize cluster separation. Additionally, the PCA plots include versions that specifically mark and highlight the top 10 most anomalous MOFs identified by the models.
