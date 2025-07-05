# Star/Galaxy Classification in Photometric Catalogs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and results for the Final Degree Project (TFG) in Physics, focusing on star/galaxy classification using machine learning techniques.

### Abstract

This work investigates the potential of machine learning (ML) for star-galaxy classification using data from the ALHAMBRA survey, which provides photometry across 20 optical medium bands and 3 near-infrared broad bands. By utilizing a matched dataset within the COSMOS field, we employ definitive morphological classifications from the COSMOS2020 catalog (based on Hubble Space Telescope imaging) as high-quality labels to train ML models. The goal is to build robust classifiers that rely only on ALHAMBRA's ground-based photometric and morphological features, enabling high-quality classification across all ALHAMBRA fields where HST data is not available.

## 📂 Repository Structure

```
javiimo-ml-for-star-galaxy-classification/
│
├── full_notebook.ipynb         # Main Jupyter Notebook with the complete workflow.
├── DEA.ipynb                   # Exploratory Data Analysis notebook (initial, less polished).
│
├── trained_models/             # Contains final, trained models and data scalers.
│   ├── group_1/                # Models trained on morphological features only.
│   ├── group_2/                # Models trained on photometric features only.
│   └── group_3/                # Models trained on combined features.
│
├── best_params/                # Optimal hyperparameters found via Hyperband search.
│   ├── group_1_best_params.json
│   ├── ...
│
├── results/                    # JSON logs of performance metrics from model runs.
│   ├── all_group_results_...
│   └── ...
│
├── raw_cvap.py                 # Custom Cross Venn-Abers Predictor script.
├── cvap_platt.py               # Custom CVAP variant using Platt Scaling.
└── LICENSE                     # MIT License.
```

## 🚀 Getting Started

### Dependencies

To run the main notebook (`full_notebook.ipynb`), you'll need Python 3.9+ and the following libraries. You can install them via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab joblib xgboost lightgbm tqdm
```

### Running the Notebook

1.  Clone the repository:
    ```bash
    git clone https://github.com/javiimo/ml-for-star-galaxy-classification.git
    cd ml-for-star-galaxy-classification
    ```
2.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
3.  Open `full_notebook.ipynb` and execute the cells. The notebook is structured to load pre-trained models if they exist, or run the full HPO and training pipeline otherwise.

## 🛠️ Methodology Overview

### Feature Groups

To assess the predictive power of different data types, the models were trained on three distinct feature sets:
*   **Group 1 (G1):** Morphological features only (e.g., `fwhm`, `stell`, `area`).
*   **Group 2 (G2):** Photometric features only (23-band magnitudes and their uncertainties).
*   **Group 3 (G3):** Combined morphological and photometric features (55 in total).

### Machine Learning Models

The study evaluates and compares several models. The tree-based models (Random Forest, XGBoost) were trained on the original feature values, while the SVM was trained on standardized (scaled) features.
*   Support Vector Machine (SVM)
*   Random Forest (RF)
*   Gradient Boosted Trees (XGBoost)

Hyperparameter optimization was performed using **Hyperband**, a modern and efficient search algorithm.

### Calibration

Reliable probability estimates are crucial. This project explored several calibration techniques:
*   **Isotonic Regression & Platt Scaling:** Standard scikit-learn methods.
*   **Cross Venn-Abers Predictor (CVAP):** A non-parametric method for generating well-calibrated probabilities. The script `raw_cvap.py` is a custom modification of the original `venn_abers.py` from the [venn-abers library](https://github.com/ip200/venn-abers/blob/main/src/venn_abers.py) to support models that output raw scores (like SVM margins) instead of just probabilities in the [0, 1] range.
*   **CVAP-Platt:** A custom-built hybrid approach (`cvap_platt.py`) that uses parametric Platt Scaling within the CVAP framework.

**Note:** While the CVAP variants were developed for robustness, they produced model collapse and the standard **Isotonic Regression** was chosen as the most stable and effective method across all calibration methods for this particular case.

## 📈 How to Use the Trained Models

The best-performing models are saved in the `trained_models/` directory. Each `.joblib` file is a bundle containing the fitted predictor and its optimized decision threshold (maximized F1 score in the validation split, balancing performance in both classes equally).

The recommended model is **XGBoost** trained on **Group 3** features and calibrated with **Isotonic Regression**.

### Recommended Usage: XGBoost Model (Group 3)

The XGBoost model was trained on **unscaled** data and does not require feature scaling for prediction.

```python
import joblib
import pandas as pd

# 1. Define the 55 feature columns for Group 3
group3_features = [
    # Morphological features (7)
    'area', 'fwhm', 'stell', 'ell', 'rk', 'rf', 's2n',
    # Photometric magnitudes (24)
    'F365W', 'F396W', 'F427W', 'F458W', 'F489W', 'F520W', 'F551W', 'F582W',
    'F613W', 'F644W', 'F675W', 'F706W', 'F737W', 'F768W', 'F799W', 'F830W',
    'F861W', 'F892W', 'F923W', 'F954W', 'J', 'H', 'KS', 'F814W',
    # Photometric uncertainties (24)
    'dF365W', 'dF396W', 'dF427W', 'dF458W', 'dF489W', 'dF520W', 'dF551W', 'dF582W',
    'dF613W', 'dF644W', 'dF675W', 'dF706W', 'dF737W', 'dF768W', 'dF799W', 'dF830W',
    'dF861W', 'dF892W', 'dF923W', 'dF954W', 'dJ', 'dH', 'dKS', 'dF814W'
]

# 2. Load the model bundle
model_path = 'trained_models/group_3/XGBoost_group_3_isotonic_cv.joblib'
bundle = joblib.load(model_path)

# Extract the model and its optimal threshold
xgb_predictor = bundle["model"]
opt_threshold = bundle["optimal_threshold"]

# 3. Prepare your new data (must have the same 55 features)
# new_data = pd.DataFrame(..., columns=group3_features) # Your data here

# 4. Get calibrated probabilities (P(star))
# proba_star = xgb_predictor.predict_proba(new_data)[:, 1]

# print(f"Optimal Threshold: {opt_threshold:.3f}")
# print(f"Probability of being a star: {proba_star}")

# 5. Get hard classifications using the optimal threshold
# predictions = (proba_star >= opt_threshold).astype(int)
# print(f"Classification (0=Galaxy, 1=Star): {predictions}")
```

### Advanced Usage: Accessing the Uncalibrated (Raw) Model

If you need the underlying, uncalibrated estimator (e.g., the raw `XGBClassifier` instance), you can extract it from one of the saved predictor objects that contain it, like the CVAP model.

```python
# Load the CVAP model to access the raw estimator inside
cvap_model_path = 'trained_models/group_3/XGBoost_group_3_cvap.joblib'
cvap_predictor = joblib.load(cvap_model_path)

# The 'final_estimator_' attribute holds the uncalibrated model
raw_xgb_model = cvap_predictor.final_estimator_

# You can now use this raw model directly (e.g., to inspect feature importances)
# print(raw_xgb_model.feature_importances_)
```

## 📓 Notebook Overview (`full_notebook.ipynb`)

The main notebook provides a comprehensive walkthrough of the entire project:

*   **Section 0: Setup and Configuration:** Imports, logging setup, and global constants.
*   **Section 1: Loading Dataset & Feature Selection:** Data loading and definition of the feature groups.
*   **Section 2: Data Preprocessing and Splitting:** Cleaning the data and splitting it into training, validation, and test sets using a stratified approach.
*   **Section 3: Implementation Details:**
    *   **Hyperparameter Optimization:** Implementation of Hyperband and other HPO techniques.
    *   **Calibration:** Code for Platt, Isotonic, and CVAP calibrators.
    *   **Metrics & Feature Scaling:** Definition of evaluation metrics and the scaling workflow.
*   **Section 4: Model Workflows:** Dedicated cells that run the full pipeline (HPO, training, calibration) for SVM, Random Forest, and XGBoost.
*   **Section 5 & 6: Model Evaluation:** Analysis and tabulation of results, comparing raw model performance against calibrated model performance using both default and optimized thresholds.

## 🏆 Results

The analysis shows that the **XGBoost** model trained on the combined morphological and photometric features (**Group 3**) and calibrated with **Isotonic Regression** consistently achieves the best performance. For detailed results, please see the final summary tables in `full_notebook.ipynb` or inspect the JSON files in the `results/` directory.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was developed as a Final Degree Project at the **Facultat de Física, Universitat de València**.

*   **Author:** Javier Montané Ortuño
*   **Tutor:** Pablo Arnalte Mur