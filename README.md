# ChemPyQSPR — Quantitative Structure-Property Relationship for Flash Point Prediction

ChemPyQSPR is an end-to-end Machine Learning and Cheminformatics framework designed for accurate Quantitative Structure-Property Relationship (QSPR) modeling of organic compound **flash points (°C)**.

The framework integrates chemical data collection & ETL, automated structure sanitization (RDKit), functional group classification, physicochemical featurization + 1024-bit Morgan Fingerprints (ECFP4), out-of-fold residual analysis, 3-stage chemical/statistical data curation, multi-model hyperparameter benchmarking (Ridge, SVR, Random Forest, XGBoost), applicability domain evaluation, and flexible inference interfaces.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Pipeline Execution Guide](#pipeline-execution-guide)
  - [Mandatory Pipeline (Data Collection to Final Training)](#mandatory-pipeline)
  - [Inference & Prediction Tools](#inference--prediction-tools)
  - [Optional Analysis Scripts](#optional-analysis-scripts)
- [Applicability Domain](#applicability-domain)
- [Data Sources & Attribution](#data-sources--attribution)
- [License](#license)

---

## Overview

Flash point is a crucial safety property indicating the lowest temperature at which a volatile liquid produces sufficient vapor to ignite in air. ChemPyQSPR automates the entire QSPR lifecycle:

1. **ETL, Sanitization & Thermal Bounds Filter**: Merges raw datasets, resolves SMILES via PubChem REST API, converts units to °C, applies organic sanitization + temperature cutoff (-150 °C to 500 °C), and deduplicates using median canonical SMILES.
2. **Structural Annotation**: Identifies main and secondary organic functional groups using RDKit SMARTS patterns.
3. **Hybrid Featurization**: Combines 10 continuous RDKit descriptors (MW, LogP, TPSA, HBD, HBA, Rotatable Bonds, Aromatic Rings, Heavy Atoms, Molar Refractivity, Fraction CSP3) with 1024-bit Morgan Fingerprints (ECFP4, radius 2).
4. **3-Stage Data Curation**: Removes radical/unstable species, flags thermodynamic unit conversion anomalies, and filters extreme residual outliers ($|z| > 3.0\sigma$) derived from Out-of-Fold (OOF) cross-validation.
5. **Model Training & Artifact Generation**: Trains candidate models (Ridge, SVR, Random Forest, XGBoost) on the curated dataset with low-variance feature filtering (`VarianceThreshold`), exporting the final champion model artifacts.
6. **Inference & Domain Validation**: Evaluates target compound applicability domain (PCA bounding box / convex envelope & inorganic flag) before executing predictions.

---

## Repository Structure

```text
ChemPyQSPR/
├── data/
│   └── raw/
│       └── molecule_features_flashpoint.csv   # Seed dataset from CHEMINFO
├── models/                                    # Generated model artifacts (git-ignored)
│   ├── best_model.pkl                         # Trained champion model pipeline
│   ├── feature_cols.pkl                       # Feature column schema
│   └── scaler.pkl                             # External scaler artifact
├── src/
│   ├── collect_flash_points.py                # 1. ETL, PubChem query, temperature filter & deduplication
│   ├── identify_funcional_groups.py           # 2. SMARTS functional group classification
│   ├── feature_engineering.py                 # 3. Descriptor & Morgan fingerprint calculator
│   ├── cross_validation_and_residuals.py      # 4. Out-of-Fold CV & residual distribution analysis
│   ├── remove_outliers.py                     # 5. 3-stage chemical & statistical curation
│   ├── train_model.py                         # 6. Champion model retraining on curated dataset & artifact export
│   ├── inference.py                           # 7a. Full inference engine (CLI, Batch CSV, REPL)
│   ├── predict_interactive.py                 # 7b. Dedicated interactive terminal REPL
│   └── analysis/                              # Optional EDA, benchmark & statistical analysis scripts
│       ├── __init__.py
│       ├── train_model_benchmark.py           # Initial grid search & benchmark on raw dataset
│       ├── exploratory_data_analysis.py       # Descriptive stats, distributions & heatmaps
│       ├── statistical_tests.py               # Kruskal-Wallis & Mann-Whitney tests
│       ├── polyfunctional_analysis.py         # Functional group co-occurrence analysis
│       ├── fingerprint_analysis.py            # PCA, t-SNE & bit correlation analysis
│       └── feature_importance.py              # SHAP & tree feature importance
├── .licenses/
│   └── pubchem_database_LICENSE.txt           # PubChem API attribution license
├── .gitignore
├── LICENSE                                    # MIT License
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python **3.10+** (Python 3.12 recommended)
- `git`

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jdumard/ChemPyQSPR.git
   cd ChemPyQSPR
   ```

2. **Create and activate a virtual environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Pipeline Execution Guide

> [!NOTE]
> Processed datasets in `data/processed/` and trained models in `models/` are generated automatically by running the pipeline steps below.

### Mandatory Pipeline

Run the steps sequentially to process the raw dataset, curate outliers, and build the champion flash point prediction model:

#### Step 1: Data Collection, ETL & Temperature Filtering
Extracts records from seed data and PubChem API, standardizes units to °C, applies chemical sanitization, enforces thermal sanity bounds (-150 °C to 500 °C), and deduplicates by median canonical SMILES.
```bash
python src/collect_flash_points.py
```
- **Inputs**: `data/raw/molecule_features_flashpoint.csv` (or remote CHEMINFO repository)
- **Output**: `data/processed/flash_point_database.csv`

#### Step 2: Functional Group Classification
Annotates chemical structures with primary and secondary functional group classes using RDKit SMARTS patterns.
```bash
python src/identify_funcional_groups.py
```
- **Input**: `data/processed/flash_point_database.csv`
- **Output**: `data/processed/cleaned_flash_point_database_fg.csv`

#### Step 3: Feature Engineering
Calculates 10 continuous RDKit physicochemical descriptors and 1024-bit Morgan Fingerprints (ECFP4).
```bash
python src/feature_engineering.py
```
- **Input**: `data/processed/cleaned_flash_point_database_fg.csv`
- **Output**: `data/processed/featurized_dataset.csv`

#### Step 4: Out-of-Fold Cross-Validation & Residual Analysis
Generates Out-of-Fold (OOF) predictions on the featurized dataset to calculate residual statistics needed for outlier filtering.
```bash
python src/cross_validation_and_residuals.py
```
- **Input**: `data/processed/featurized_dataset.csv`
- **Output**: `data/processed/residual_analysis/cv_predictions_and_residuals.csv`

#### Step 5: 3-Stage Data Curation & Outlier Removal
Applies 3 decoupled filters: (1) Unstable species/radicals, (2) Thermodynamic unit/MW anomalies, and (3) $|z| > 3.0\sigma$ residual outliers from cross-validation.
```bash
python src/remove_outliers.py
```
- **Inputs**:
  - `data/processed/featurized_dataset.csv`
  - `data/processed/cleaned_flash_point_database_fg.csv`
  - `data/processed/residual_analysis/cv_predictions_and_residuals.csv`
- **Outputs**:
  - `data/processed/featurized_dataset_curated.csv`
  - `data/processed/cleaned_flash_point_database_curated_fg.csv`
  - `data/processed/curation_results/`

#### Step 6: Final Model Training & Artifact Generation
Retrains model candidates (Ridge, SVR, RF, XGBoost) on the curated dataset, applies variance thresholding, and exports the final champion model artifacts.
```bash
python src/train_model.py
```
- **Input**: `data/processed/featurized_dataset_curated.csv`
- **Outputs**:
  - Champion artifacts in `models/` (`best_model.pkl`, `feature_cols.pkl`, `scaler.pkl`)
  - `data/processed/model_results_reduced_curated/`

---

### Inference & Prediction Tools

Once the model artifacts are trained and available in `models/`, use either inference interface:

#### Interactive Terminal REPL (Recommended)
Interactive session supporting compound name lookup via PubChem or direct SMILES input (prefixed with `s:`).
```bash
python src/predict_interactive.py
```

#### Command-Line Interface (CLI)
Predict flash points directly from command-line arguments:
```bash
# By compound names (queries PubChem automatically):
python src/inference.py "ethanol" "acetone" "benzene"

# Direct SMILES strings:
python src/inference.py "CCO" "CC(=O)C" "c1ccccc1" --smiles
```

#### Batch Inference (CSV Files)
Process an input CSV containing compound names or SMILES, appending predicted flash points and applicability domain warnings:
```bash
python src/inference.py --input-csv data/sample_inputs.csv --output-csv data/sample_predictions.csv
```

---

### Optional Analysis Scripts

Scripts located in `src/analysis/` generate auxiliary diagnostic reports, visualizations, and interpretability metrics:

- **Preliminary Grid Search & Algorithm Benchmark**:
  ```bash
  python src/analysis/train_model_benchmark.py
  ```
  Runs initial hyperparameter grid search on the raw uncurated dataset to benchmark algorithms.

- **Exploratory Data Analysis (EDA)**:
  ```bash
  python src/analysis/exploratory_data_analysis.py
  ```
  Generates target distributions, boxplots by functional group, and descriptor correlation heatmaps in `data/processed/eda_results/`.

- **Statistical Significance Tests**:
  ```bash
  python src/analysis/statistical_tests.py
  ```
  Applies non-parametric Kruskal-Wallis and Bonferroni-corrected Mann-Whitney U tests across chemical families in `data/processed/statistical_results/`.

- **Polyfunctional Co-occurrence Analysis**:
  ```bash
  python src/analysis/polyfunctional_analysis.py
  ```
  Analyzes functional group co-occurrence matrices and impact on flash point variance in `data/processed/polyfunctional_results/`.

- **Morgan Fingerprint Density & Dimensionality Reduction**:
  ```bash
  python src/analysis/fingerprint_analysis.py
  ```
  Executes PCA and t-SNE projections alongside point-biserial bit correlation analysis in `data/processed/fingerprint_results/`.

- **Feature Importance & SHAP Interpretability**:
  ```bash
  python src/analysis/feature_importance.py
  ```
  Computes SHAP (SHapley Additive exPlanations) values and MDI feature importances for XGBoost in `data/processed/feature_importance/`.

---

## Applicability Domain

To ensure prediction reliability, every inference query is validated against an **Applicability Domain (AD)** filter:

1. **Inorganic Check**: Compounds lacking carbon-carbon / carbon-hydrogen organic frameworks (e.g., metals, salts) are flagged as out of domain and blocked.
2. **Physicochemical Envelope**: Descriptor ranges (MW, LogP, TPSA, etc.) and Morgan Fingerprint density are evaluated against the training set envelope.
3. **Warning Feedback**: Out-of-domain predictions are flagged with descriptive status warnings (e.g., `OUT OF DOMAIN WARNING`), allowing users to inspect confidence boundaries.

---

## Data Sources & Attribution

- **CHEMINFO Dataset**: Initial flash point seed data compiled from open QSPR benchmark datasets ([molecule-features repository](https://github.com/cheminfo/molecule-features)).
- **PubChem REST API**: Molecular identifiers and SMILES strings resolved via NCBI PubChem PUG REST and PUG View services. See [.licenses/pubchem_database_LICENSE.txt](.licenses/pubchem_database_LICENSE.txt) for usage terms.

---

## License

This project is licensed under the [MIT License](LICENSE).
