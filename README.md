# WiFi Indoor Positioning — Machine Learning Project

## Project Overview

This project explores indoor positioning using WiFi RSSI fingerprints from the UJIIndoorLoc dataset. GPS is not dependable inside buildings, so the system uses the signal-strength pattern measured from nearby WiFi access points as a location fingerprint.

Several machine-learning techniques are studied, including PCA, kNN, SVM, Decision Trees, and Gradient Boosting. The experiments also examine training-set size, PCA feature count, model parameters, missing WiFi observations, model combinations, and final validation on unseen data.

---

## Dataset

The project uses the **UJIIndoorLoc** dataset. Each WiFi fingerprint contains **520 WiFi access-point RSSI features**, along with building and floor information.

Dataset source:

https://archive.ics.uci.edu/dataset/310/ujiindoorloc

Place the two required files here:

```text
data/
├── trainingData.csv
└── validationData.csv
```

The large CSV files are intentionally not committed to Git.

In the original data, RSSI value `100` means that an access point was not detected. During preprocessing, this is changed to `-105`, representing a very weak signal.

---

## System Workflow

The complete project follows this general flow:

```text
WiFi Dataset
    ↓
Preprocessing
    ↓
PCA / Feature Processing
    ↓
Machine Learning Experiments
    ↓
Cross-Validation
    ↓
Model Comparison
    ↓
Final Validation
    ↓
Two-Stage Building + Floor Prediction
    ↓
Streamlit Demonstration
```

The final positioning pipeline is hierarchical:

```text
WiFi RSSI fingerprint
        ↓
Building prediction
        ↓
Select the corresponding building floor model
        ↓
Floor prediction
        ↓
Final Building + Floor
```

---

## Algorithms

| Technique | Main use |
|---|---|
| PCA | Reduce the dimensionality of WiFi features |
| kNN | Fingerprint-based classification |
| SVM | Compare linear/kernel classification approaches |
| Decision Tree | Tree-based classification and pruning experiments |
| Gradient Boosting | Ensemble classification using sequential trees |

The project also includes NumPy implementations of PCA and kNN created from scratch.

---

## Repository Structure

```text
indoor-positioning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── trainingData.csv
│   └── validationData.csv
│
├── src/
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── pca_from_scratch.py
│   └── tree_model.py
│
├── experiments/
│   ├── 01_explore_data.py
│   ├── 02_test_preprocessing.py
│   ├── 03_pca_analysis.py
│   ├── 04_compare_pca.py
│   ├── 05_knn_baseline.py
│   ├── 06_knn_repeated_cv.py
│   ├── 07_knn_vs_k.py
│   ├── 08_knn_feature_count.py
│   ├── 09_svm_experiments.py
│   ├── 10_decision_tree.py
│   ├── 11_gradient_boosting.py
│   ├── 12_model_comparison.py
│   ├── 13_missing_signal.py
│   └── 14_final_test.py
│
├── results/
│   ├── figures/
│   └── tables/
│
└── demo/
    ├── predict_location.py
    └── streamlit_app.py
```

The folders separate reusable source code, individual experiments, saved outputs, and the final demonstrations.

---

## Requirements

The implementation uses Python with the following major libraries:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- joblib
- Streamlit

The exact package versions are listed in `requirements.txt`.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd indoor-positioning
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Setup

Make sure the project contains:

```text
indoor-positioning/
└── data/
    ├── trainingData.csv
    └── validationData.csv
```

All experiment commands should be executed from the project root.

---

## Experiments

The experiment scripts are intended to be run in sequence when reproducing the analysis.

### Dataset inspection

```bash
python experiments/01_explore_data.py
```

### Preprocessing check

```bash
python experiments/02_test_preprocessing.py
```

### PCA analysis

```bash
python experiments/03_pca_analysis.py
```

### PCA comparison

```bash
python experiments/04_compare_pca.py
```

### kNN baseline

```bash
python experiments/05_knn_baseline.py
```

### Repeated kNN cross-validation

```bash
python experiments/06_knn_repeated_cv.py
```

### kNN parameter experiment

```bash
python experiments/07_knn_vs_k.py
```

### Feature-count experiment

```bash
python experiments/08_knn_feature_count.py
```

### SVM

```bash
python experiments/09_svm_experiments.py
```

### Decision Tree

```bash
python experiments/10_decision_tree.py
```

### Gradient Boosting

```bash
python experiments/11_gradient_boosting.py
```

### Model comparison

```bash
python experiments/12_model_comparison.py
```

### Missing-signal experiment

```bash
python experiments/13_missing_signal.py
```

### Final validation

```bash
python experiments/14_final_test.py
```

The generated plots and numerical tables are saved under `results/`.

---

## Command-Line Demo

The command-line demonstration can be started with:

```bash
python demo/predict_location.py
```

The program first predicts the building from the WiFi fingerprint. It then uses the floor model associated with that predicted building to obtain the floor. For validation samples, the prediction can be compared with the known building and floor.

---

## Streamlit Demo

The interactive application is launched using:

```bash
streamlit run demo/streamlit_app.py
```

The UI lets the user select a validation fingerprint and run it through the final positioning pipeline.

The application can display:

- selected sample
- reference building and floor
- WiFi fingerprint information
- predicted building
- predicted floor
- available model scores
- actual versus predicted location
- building, floor, and end-to-end correctness

The Streamlit application is the presentation layer for the final model pipeline.

---

## How the Final Model Works

### Stage 1 — Building

The WiFi RSSI fingerprint is given to the building classifier. The project uses **k-Nearest Neighbors** for this stage, where nearby training fingerprints influence the predicted building.

### Stage 2 — Floor

Once the building is identified, the same fingerprint is sent to the corresponding building-specific floor classifier. The project uses **Gradient Boosting** models for these floor predictions.

Therefore, the final result is:

```text
RSSI fingerprint
      ↓
Building
      ↓
Building-specific floor model
      ↓
Floor
```

---

## Reported Results

The recorded model comparison at 5,000 training samples is:

| Model | Accuracy | Error |
|---|---:|---:|
| kNN | ~90% | ~10% |
| SVM | ~84% | ~16% |
| Decision Tree | ~91% | ~9% |
| Gradient Boosting | ~96.64% | ~3.36% |

The final two-stage validation reported:

| Metric | Accuracy |
|---|---:|
| Building prediction | 99.19% |
| Floor prediction | 89.47% |
| End-to-end prediction | 89.47% |

The results indicate that building classification is highly accurate, while floor-level classification is more challenging and therefore limits the final end-to-end performance.

---

## Results Directory

### Figures

`results/figures/` contains visual outputs covering topics such as:

- PCA variance/energy
- Decision Tree pruning
- Gradient Boosting iterations
- feature importance
- per-floor Gradient Boosting performance
- missing-signal robustness

### Tables

`results/tables/` stores numerical experiment outputs for:

- model comparisons
- SVM
- Decision Trees
- Gradient Boosting
- feature importance
- per-floor results
- final validation
- model combinations
- missing-signal analysis

---

## Main Conclusions

1. WiFi RSSI fingerprints can be used for indoor positioning without GPS.
2. Increasing the available training data generally improves the evaluated models.
3. Gradient Boosting gives strong results, especially with larger training sets.
4. Building identification is easier than distinguishing individual floors.
5. Missing WiFi measurements can reduce prediction performance.
6. Separating building prediction from floor prediction provides a practical two-stage positioning pipeline.

---

## Team

- **Member 1:** Chinmay N S
- **SRN 1:** PES2UG24AM047
- **Member 2:** KOUSHIK
- **SRN 2:** PES2UG24AM076
