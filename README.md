# Indoor Positioning Using WiFi Fingerprints

## Project Overview

This project investigates indoor positioning using WiFi RSSI fingerprints from the UJIIndoorLoc dataset.

GPS is unreliable indoors, while WiFi access points are commonly available. Each location produces a different pattern of received signal strengths (RSSI), which can be treated as a WiFi fingerprint.

The project compares multiple machine learning approaches for indoor floor prediction:

- Principal Component Analysis (PCA)
- k-Nearest Neighbors (kNN)
- Support Vector Machine (SVM)
- Decision Tree
- Gradient Boosting

The project also studies the effect of training sample size, PCA feature count, model parameters, missing WiFi signals, model combinations, and final validation on unseen data.

---

## Dataset

The project uses the **UJIIndoorLoc** dataset.

The dataset contains WiFi RSSI fingerprints collected from multiple buildings and floors. Each fingerprint contains 520 WiFi access-point signal-strength features.

### Dataset files

Download the UJIIndoorLoc dataset from the official UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/310/ujiindoorloc

Place the following files inside the `data/` directory:

```text
data/
├── trainingData.csv
└── validationData.csv
```

The dataset files are intentionally excluded from Git because they are large and are not required to be committed to the repository.

The value `100` in the original dataset represents a WiFi access point that was not detected. During preprocessing, these values are replaced with `-105` so that an undetected signal is represented as a very weak RSSI value.

---

## Approach

The project follows this pipeline:

```text
Raw WiFi RSSI Data
        ↓
Data Cleaning & Preprocessing
        ↓
PCA / Feature Reduction
        ↓
Machine Learning Models
        ↓
Repeated Cross-Validation
        ↓
Model Comparison
        ↓
Final Validation
        ↓
Two-Stage Building + Floor Prediction
        ↓
Streamlit Demo
```

### Models evaluated

| Model | Purpose |
|---|---|
| kNN | Distance-based WiFi fingerprint classification |
| SVM | Linear and kernel-based classification |
| Decision Tree | Tree-based classification with pruning |
| Gradient Boosting | Ensemble model using sequential decision trees |

PCA is used to reduce the dimensionality of the WiFi fingerprint features.

kNN and PCA implementations were also developed from scratch using NumPy as part of the project.

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

---

## Requirements

The project uses Python and the following main packages:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- joblib
- Streamlit

Exact package versions are provided in `requirements.txt`.

---

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd indoor-positioning
```

Create and activate a virtual environment.

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

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Create the data directory if it does not already exist:

```text
data/
```

Place:

```text
trainingData.csv
validationData.csv
```

inside the `data/` directory.

The expected structure is:

```text
indoor-positioning/
└── data/
    ├── trainingData.csv
    └── validationData.csv
```

---

## Running the Experiments

All experiments should be run from the project root.

### 1. Explore the dataset

```bash
python experiments/01_explore_data.py
```

### 2. Test preprocessing

```bash
python experiments/02_test_preprocessing.py
```

### 3. PCA analysis

```bash
python experiments/03_pca_analysis.py
```

### 4. Compare PCA implementations

```bash
python experiments/04_compare_pca.py
```

### 5. kNN baseline

```bash
python experiments/05_knn_baseline.py
```

### 6. Repeated cross-validation for kNN

```bash
python experiments/06_knn_repeated_cv.py
```

### 7. kNN parameter study

```bash
python experiments/07_knn_vs_k.py
```

### 8. kNN feature-count study

```bash
python experiments/08_knn_feature_count.py
```

### 9. SVM experiments

```bash
python experiments/09_svm_experiments.py
```

### 10. Decision Tree experiments

```bash
python experiments/10_decision_tree.py
```

### 11. Gradient Boosting experiments

```bash
python experiments/11_gradient_boosting.py
```

### 12. Model comparison

```bash
python experiments/12_model_comparison.py
```

### 13. Missing-signal robustness experiment

```bash
python experiments/13_missing_signal.py
```

### 14. Final validation

```bash
python experiments/14_final_test.py
```

Generated figures and tables are stored in:

```text
results/
```

---

## Running the Command-Line Demo

The project includes a command-line location prediction demo.

Run:

```bash
python demo/predict_location.py
```

The demo performs two-stage prediction:

1. Predict the building.
2. Predict the floor within the predicted building.

It reports the predicted location together with the actual location for validation examples.

---

## Running the Streamlit Demo

The project also includes an interactive Streamlit application.

From the project root, run:

```bash
streamlit run demo/streamlit_app.py
```

The application provides an interactive interface for the indoor positioning system and demonstrates the final prediction pipeline.

---

## Results

### Model Comparison

The experiments show that model performance generally improves as the amount of training data increases.

Gradient Boosting performs particularly strongly at larger training sample sizes, while kNN also provides strong performance. SVM performs comparatively worse in the evaluated experiments.

At 5,000 training samples, the recorded results include:

| Model | Accuracy | Error |
|---|---:|---:|
| kNN | ~90% | ~10% |
| SVM | ~84% | ~16% |
| Decision Tree | ~91% | ~9% |
| Gradient Boosting | ~96.64% | ~3.36% |

### Final Two-Stage Validation

The final system first predicts the building and then predicts the floor within the predicted building.

| Metric | Accuracy |
|---|---:|
| Building prediction | 99.19% |
| Floor prediction | 89.47% |
| End-to-end prediction | 89.47% |

The final validation also shows differences in floor-prediction performance between buildings, demonstrating that some buildings are more difficult to classify than others.

---

## Results and Outputs

The repository contains the generated results from the experiments.

### Figures

```text
results/figures/
```

contains plots for:

- PCA variance / energy
- Decision Tree pruning
- Gradient Boosting iterations
- Gradient Boosting feature importance
- Gradient Boosting per-floor performance
- Missing-signal robustness

### Tables

```text
results/tables/
```

contains the numerical results from:

- Model comparison
- SVM experiments
- Decision Tree experiments
- Gradient Boosting experiments
- Feature importance
- Per-floor performance
- Final validation
- Model combinations
- Missing-signal experiments

---

## Key Findings

1. WiFi RSSI fingerprints can be used effectively for indoor floor prediction.
2. Increasing the amount of training data generally improves model performance.
3. Gradient Boosting provides strong performance, particularly with larger training datasets.
4. The building classifier achieves very high accuracy, while the more difficult floor-level prediction determines the final end-to-end accuracy.
5. Missing WiFi signals can reduce prediction performance, making robustness to incomplete fingerprints an important consideration for real-world deployment.

---

## Team

- **Member 1:** Chinmay N S
- **Member 2:** KOUSHIK