import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import joblib # For loading scaler if it exists

# --- Setup Logging ---
LOG_FILENAME = f'investigation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    force=True
)
# Prevent logs from being printed to console, direct them only to file
# logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
print(f"Investigation script started. Logging details to: {LOG_FILENAME}")


# --- Configuration ---
DATA_FILE = 'data/match_alhambra_cosmos2020_ACS_class_0.8arcsec.csv'
BEST_PARAMS_DIR = "best_params"
MODEL_DIR = "trained_models" # Used by apply_feature_scaling to potentially load scalers
TARGET_COLUMN = 'acs_mu_class'
RANDOM_SEED = 33

GROUPS_TO_INVESTIGATE = ['group_2', 'group_3', 'group_6']

# --- Global Variables for Data Splitting (from Model.txt) ---
TEST_SIZE = 0.20
VAL_SIZE = 0.10
CAL_SIZE = 0.10
SPLIT_STRATEGY = 'stratified'


# --- Feature Set Definitions (Copied from Model.txt [In[5]]) ---
morphology_features = [
    'area', 'fwhm', 'stell', 'ell', 'a', 'b', 'theta', 'rk', 'rf'
]
morphology_err = ['s2n']
morphology_mags_errors = morphology_features + morphology_err

OPTICAL_MAG_COLS = [
    'F365W', 'F396W', 'F427W', 'F458W', 'F489W', 'F520W', 'F551W',
    'F582W', 'F613W', 'F644W', 'F675W', 'F706W', 'F737W', 'F768W',
    'F799W', 'F830W', 'F861W', 'F892W', 'F923W', 'F954W'
]
photometry_magnitudes = OPTICAL_MAG_COLS + ['J', 'H', 'KS', 'F814W']

OPTICAL_ERR_COLS = [
    'dF365W', 'dF396W', 'dF427W', 'dF458W', 'dF489W', 'dF520W', 'dF551W',
    'dF582W', 'dF613W', 'dF644W', 'dF675W', 'dF706W', 'dF737W', 'dF768W',
    'dF799W', 'dF830W', 'dF861W', 'dF892W', 'dF923W', 'dF954W'
]
photometry_uncertainties = OPTICAL_ERR_COLS + ['dJ', 'dH', 'dKS', 'dF814W']
photometry_mags_errors = photometry_magnitudes + photometry_uncertainties

redshift_features = [
    'zb_1', 'zb_Min_1', 'zb_Max_1', 'Tb_1',
    'z_ml', 't_ml',
    'Stell_Mass_1', 'M_Abs_1', 'MagPrior'
]
redshift_uncertainties = ['Odds_1', 'Chi2']
redshift_mags_errors = redshift_features + redshift_uncertainties

OPTICAL_IRMS_COLS = [
    'irms_F365W', 'irms_F396W', 'irms_F427W', 'irms_F458W', 'irms_F489W',
    'irms_F520W', 'irms_F551W', 'irms_F582W', 'irms_F613W', 'irms_F644W',
    'irms_F675W', 'irms_F706W', 'irms_F737W', 'irms_F768W', 'irms_F799W',
    'irms_F830W', 'irms_F861W', 'irms_F892W', 'irms_F923W', 'irms_F954W'
]
quality_aux_features = ['nfobs'] + OPTICAL_IRMS_COLS + ['irms_J', 'irms_H', 'irms_KS', 'irms_F814W']

target_variable_list = ['acs_mu_class']

feature_sets_dict = {
    'morphology_only': morphology_mags_errors,
    'photometry_magnitudes_only': photometry_magnitudes,
    'photometry_mags_errors': photometry_mags_errors,
    'redshift_only': redshift_mags_errors,
    'photometry_plus_morphology': photometry_mags_errors + morphology_mags_errors,
    'full_alhambra_all': (morphology_mags_errors + photometry_mags_errors +
                           redshift_mags_errors + quality_aux_features),
    'target_variable': target_variable_list
}

groups_dict = {
    'group_1': feature_sets_dict.get('morphology_only', []) + feature_sets_dict.get('target_variable', []),
    'group_2': feature_sets_dict.get('photometry_magnitudes_only', []) + feature_sets_dict.get('target_variable', []),
    'group_3': feature_sets_dict.get('photometry_mags_errors', []) + feature_sets_dict.get('target_variable', []),
    'group_4': feature_sets_dict.get('redshift_only', []) + feature_sets_dict.get('target_variable', []),
    'group_5': feature_sets_dict.get('photometry_plus_morphology', []) + feature_sets_dict.get('target_variable', []),
    'group_6': (feature_sets_dict.get('photometry_mags_errors', []) +
               feature_sets_dict.get('morphology_only', []) + # Corrected: was morphology_features
               feature_sets_dict.get('redshift_only', []) +   # Corrected: was redshift_features
               feature_sets_dict.get('target_variable', [])),
    'group_7': feature_sets_dict.get('full_alhambra_all', []) + feature_sets_dict.get('target_variable', [])
}


# --- Helper Functions (Copied and adapted from Model.txt) ---

def get_feature_set(df_in, set_name, groups_map=groups_dict):
    # Simplified version for this script, original is in Model.txt
    if set_name not in groups_map:
        raise ValueError(f"Feature set group '{set_name}' not defined. "
                         f"Available groups: {list(groups_map.keys())}")
    required_cols_in_set = groups_map[set_name]
    available_cols = [col for col in required_cols_in_set if col in df_in.columns]
    missing_cols = [col for col in required_cols_in_set if col not in available_cols]
    if missing_cols:
        logging.warning(f"Warning for group '{set_name}': Columns defined but not found: {missing_cols}")
    if not available_cols:
        logging.warning(f"Warning for group '{set_name}': No columns found.")
        return pd.DataFrame()
    logging.debug(f"Selected feature set group '{set_name}' with {len(available_cols)} columns.")
    return df_in[available_cols]

def clean_data(df_in, feature_group, target_col_name, logger=logging):
    logger.info(f"Original dataset size: {df_in.shape} for group {feature_group}")
    df_clean = get_feature_set(df_in, feature_group, groups_map=groups_dict).dropna().copy()
    logger.info(f"Dataset size after dropping NaNs for group {feature_group}: {df_clean.shape}")

    if target_col_name not in df_clean.columns:
        raise KeyError(f"Target column '{target_col_name}' not found in the cleaned DataFrame columns: {df_clean.columns.tolist()}")
    logger.info(f"Value counts for target in group {feature_group}:\nStar (1): {(df_clean[target_col_name] == 1).sum()}, Galaxy (0): {(df_clean[target_col_name] == 0).sum()}")

    X_out = df_clean.drop(columns=[target_col_name])
    y_out = df_clean[target_col_name]
    return X_out, y_out, df_clean

def split_data(X_in, y_in): # Uses global TEST_SIZE, VAL_SIZE, CAL_SIZE, SPLIT_STRATEGY, RANDOM_SEED
    logging.info(f"Splitting data using '{SPLIT_STRATEGY}' strategy...")
    if not (0 <= TEST_SIZE <= 1 and 0 <= VAL_SIZE <= 1 and 0 <= CAL_SIZE <= 1):
        raise ValueError("Split proportions must be between 0 and 1.")
    TRAIN_SIZE = 1.0 - TEST_SIZE - VAL_SIZE - CAL_SIZE
    if not (0 <= TRAIN_SIZE <= 1):
        raise ValueError(f"Calculated TRAIN_SIZE ({TRAIN_SIZE:.3f}) is invalid.")

    empty_X = X_in.iloc[0:0]
    empty_y = y_in.iloc[0:0]
    X_train, y_train = empty_X.copy(), empty_y.copy()
    X_val, y_val = empty_X.copy(), empty_y.copy()
    X_test, y_test = empty_X.copy(), empty_y.copy()
    X_cal, y_cal = empty_X.copy(), empty_y.copy()

    X_remaining, y_remaining = X_in.copy(), y_in.copy()

    def get_stratify_array(y_arr):
        return y_arr if SPLIT_STRATEGY == 'stratified' and not y_arr.empty else None

    val_test_cal_size = VAL_SIZE + TEST_SIZE + CAL_SIZE
    if np.isclose(val_test_cal_size, 0):
        X_train, y_train = X_remaining, y_remaining
        X_remaining, y_remaining = empty_X.copy(), empty_y.copy()
    elif not np.isclose(TRAIN_SIZE, 0):
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X_remaining, y_remaining, test_size=val_test_cal_size,
            random_state=RANDOM_SEED, stratify=get_stratify_array(y_remaining)
        )
    logging.debug(f"Train set shape: {X_train.shape}")

    if not X_remaining.empty:
        test_cal_size = TEST_SIZE + CAL_SIZE
        current_remaining_size_frac = VAL_SIZE + test_cal_size
        if np.isclose(VAL_SIZE, 0):
            X_temp2, y_temp2 = X_remaining, y_remaining
        elif np.isclose(test_cal_size, 0):
            X_val, y_val = X_remaining, y_remaining
            X_temp2, y_temp2 = empty_X.copy(), empty_y.copy()
        else:
            split_test_size_val = test_cal_size / current_remaining_size_frac
            X_val, X_temp2, y_val, y_temp2 = train_test_split(
                X_remaining, y_remaining, test_size=split_test_size_val,
                random_state=RANDOM_SEED, stratify=get_stratify_array(y_remaining)
            )
        logging.debug(f"Validation set shape: {X_val.shape}")
    else:
        X_temp2, y_temp2 = empty_X.copy(), empty_y.copy()

    if not X_temp2.empty:
        current_remaining_size_frac_test = TEST_SIZE + CAL_SIZE
        if np.isclose(CAL_SIZE, 0):
            X_test, y_test = X_temp2, y_temp2
        elif np.isclose(TEST_SIZE, 0):
            X_cal, y_cal = X_temp2, y_temp2
        else:
            split_test_size_cal = CAL_SIZE / current_remaining_size_frac_test
            X_test, X_cal, y_test, y_cal = train_test_split(
                X_temp2, y_temp2, test_size=split_test_size_cal,
                random_state=RANDOM_SEED, stratify=get_stratify_array(y_temp2)
            )
    logging.debug(f"Test set shape: {X_test.shape}")
    logging.debug(f"Calibration set shape: {X_cal.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, X_cal, y_cal

def apply_feature_scaling(
    X_train, X_val, X_test, X_cal,
    # These props are for internal logic/logging, not for subsetting here
    _train_prop, _val_prop, _test_prop, _cal_prop,
    model_dir_path,
    save_scaler=True,
    group_name_str=None
):
    scaler = None
    scaler_loaded = False
    scaler_filename = None

    if group_name_str is not None:
        scaler_filename = os.path.join(model_dir_path, group_name_str, f"scaler_{group_name_str}.joblib") # Scalers might be in group subdirs
        if not os.path.exists(scaler_filename): # Fallback to old main dir scaler
             scaler_filename = os.path.join(model_dir_path, f"scaler_{group_name_str}.joblib")

        if scaler_filename and os.path.exists(scaler_filename):
            try:
                scaler = joblib.load(scaler_filename)
                scaler_loaded = True
                logging.info(f"Scaler loaded from {scaler_filename}")
            except Exception as e:
                logging.warning(f"Failed to load scaler from {scaler_filename}: {e}. Will fit a new one.")
                scaler_loaded = False # Ensure it's false
        else:
            logging.info(f"No existing scaler found for group '{group_name_str}', will fit a new one if possible.")

    X_train_scaled, X_val_scaled, X_test_scaled, X_cal_scaled = \
        X_train.copy(), X_val.copy(), X_test.copy(), X_cal.copy() # Start with copies

    if not scaler_loaded:
        if not X_train.empty:
            logging.info("Fitting a new StandardScaler on training data...")
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            if save_scaler and scaler_filename: # Save if new and filename is determined
                os.makedirs(os.path.dirname(scaler_filename), exist_ok=True)
                joblib.dump(scaler, scaler_filename)
                logging.info(f"New scaler saved to {scaler_filename}")
        else:
            logging.warning("Empty training set, cannot fit or apply StandardScaler!")
    
    if scaler: # If scaler is available (loaded or newly fitted)
        if not X_train.empty and scaler_loaded: # If loaded, transform train
             X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        if not X_val.empty:
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        if not X_test.empty:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        if not X_cal.empty:
            X_cal_scaled = pd.DataFrame(scaler.transform(X_cal), columns=X_cal.columns, index=X_cal.index)
    
    logging.info("Feature scaling processing complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, X_cal_scaled, scaler

# --- Main Investigation Function ---
def investigate():
    logging.info("Starting SVM performance investigation.")

    try:
        df_main = pd.read_csv(DATA_FILE)
        logging.info(f"Full DataFrame loaded: {df_main.shape}")
        # Preprocessing from Model.txt
        n_fakes_main = (df_main[TARGET_COLUMN] == 3).sum()
        logging.info(f"Number of fake detections (class 3): {n_fakes_main}")
        df_main = df_main[df_main[TARGET_COLUMN] != 3].copy() # Use .copy()
        df_main[TARGET_COLUMN] = df_main[TARGET_COLUMN].map({1: 0, 2: 1})
        logging.info(f"DataFrame after dropping fakes and mapping classes: {df_main.shape}")
    except FileNotFoundError:
        logging.error(f"Data file {DATA_FILE} not found. Exiting.")
        print(f"ERROR: Data file {DATA_FILE} not found. Exiting.")
        return
    except Exception as e_load:
        logging.error(f"Error loading or preprocessing data: {e_load}", exc_info=True)
        print(f"ERROR: Error loading or preprocessing data: {e_load}")
        return

    for group_name_iter in GROUPS_TO_INVESTIGATE:
        logging.info(f"\n{'='*20} Investigating Group: {group_name_iter} {'='*20}")
        print(f"\nInvestigating Group: {group_name_iter}")

        # --- Load Best SVM Params for this group ---
        svm_params_from_file = None
        params_file = os.path.join(BEST_PARAMS_DIR, f"{group_name_iter}_best_params.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f_params:
                    all_grp_params = json.load(f_params)
                    svm_params_from_file = all_grp_params.get("SVM") # Your model name
                    if svm_params_from_file:
                        logging.info(f"[{group_name_iter}] Loaded SVM params: {svm_params_from_file}")
                        svm_params_from_file.setdefault('random_state', RANDOM_SEED)
                        svm_params_from_file.setdefault('class_weight', 'balanced')
                        if 'probability' in svm_params_from_file: # As per original workflow logic
                            del svm_params_from_file['probability']
                    else:
                        logging.warning(f"[{group_name_iter}] 'SVM' parameters not found in {params_file}")
            except Exception as e_param:
                logging.error(f"[{group_name_iter}] Error loading params from {params_file}: {e_param}", exc_info=True)
        else:
            logging.warning(f"[{group_name_iter}] Best params file not found: {params_file}.")

        # --- Data Prep ---
        try:
            X_grp, y_grp, _ = clean_data(df_main.copy(), group_name_iter, TARGET_COLUMN, logger=logging)
            if X_grp.empty or y_grp.empty:
                logging.warning(f"[{group_name_iter}] Skipping group due to insufficient data after cleaning (X: {X_grp.shape}).")
                continue
        except Exception as e_clean:
            logging.error(f"[{group_name_iter}] Error during clean_data: {e_clean}", exc_info=True)
            continue

        # HYPOTHESIS 1: Number of Samples
        logging.info(f"--- HYPOTHESIS 1: Sample Size (after clean_data) ---")
        logging.info(f"[{group_name_iter}] Shape of X: {X_grp.shape}, y: {y_grp.shape}")
        if not y_grp.empty:
            logging.info(f"[{group_name_iter}] Class distribution in y: Star (1): {(y_grp == 1).sum()}, Galaxy (0): {(y_grp == 0).sum()}")

        try:
            X_train_grp, y_train_grp, X_val_grp, y_val_grp, X_test_grp, y_test_grp, X_cal_grp, y_cal_grp = split_data(X_grp, y_grp)
            if X_train_grp.empty:
                logging.warning(f"[{group_name_iter}] Training set is empty after split. Skipping model fitting tests.")
                # Continue with other data-only tests if applicable, or skip group.
            else: # Proceed with scaling and model tests only if X_train_grp is not empty
                # --- Feature Scaling ---
                # Calculate actual proportions for apply_feature_scaling's internal logic/logging
                len_X_grp = len(X_grp)
                p_train = len(X_train_grp) / len_X_grp if len_X_grp > 0 else 0
                p_val   = len(X_val_grp)   / len_X_grp if len_X_grp > 0 else 0
                p_test  = len(X_test_grp)  / len_X_grp if len_X_grp > 0 else 0
                p_cal   = len(X_cal_grp)   / len_X_grp if len_X_grp > 0 else 0

                X_train_scl, _, _, _, _ = apply_feature_scaling(
                    X_train_grp, X_val_grp, X_test_grp, X_cal_grp,
                    p_train, p_val, p_test, p_cal,
                    MODEL_DIR, save_scaler=False, group_name_str=group_name_iter
                ) # Only need X_train_scl for these tests

                # HYPOTHESIS 2: Number of Support Vectors & Simpler Kernel
                logging.info(f"--- HYPOTHESIS 2: Support Vectors & Simpler Kernel (on X_train_scl) ---")
                if svm_params_from_file and svm_params_from_file.get('kernel') == 'poly':
                    try:
                        logging.info(f"[{group_name_iter}] Testing polynomial kernel: {svm_params_from_file}")
                        svm_poly_model = SVC(**svm_params_from_file)
                        start_t_poly = time.time()
                        svm_poly_model.fit(X_train_scl, y_train_grp)
                        fit_t_poly = time.time() - start_t_poly
                        logging.info(f"[{group_name_iter}] Poly SVM fit time: {fit_t_poly:.2f}s")
                        logging.info(f"[{group_name_iter}] Poly SVM n_support_ (per class): {svm_poly_model.n_support_}, total: {np.sum(svm_poly_model.n_support_)}")
                    except Exception as e_poly_svm:
                        logging.error(f"[{group_name_iter}] Error fitting polynomial SVM: {e_poly_svm}", exc_info=True)
                else:
                    logging.warning(f"[{group_name_iter}] Skipping polynomial SVM test (params missing, not poly, or X_train_scl empty).")

                # Test Linear Kernel
                try:
                    c_val = svm_params_from_file.get('C', 1.0) if svm_params_from_file else 1.0
                    svm_linear_p = {'kernel': 'linear', 'C': c_val, 'random_state': RANDOM_SEED, 'class_weight': 'balanced'}
                    logging.info(f"[{group_name_iter}] Testing linear kernel: {svm_linear_p}")
                    svm_linear_model = SVC(**svm_linear_p)
                    start_t_linear = time.time()
                    svm_linear_model.fit(X_train_scl, y_train_grp)
                    fit_t_linear = time.time() - start_t_linear
                    logging.info(f"[{group_name_iter}] Linear SVM fit time: {fit_t_linear:.2f}s")
                    logging.info(f"[{group_name_iter}] Linear SVM n_support_ (per class): {svm_linear_model.n_support_}, total: {np.sum(svm_linear_model.n_support_)}")
                except Exception as e_linear_svm:
                    logging.error(f"[{group_name_iter}] Error fitting linear SVM: {e_linear_svm}", exc_info=True)

                # HYPOTHESIS 3: Numerical Properties of Scaled Features
                logging.info(f"--- HYPOTHESIS 3: Numerical Properties of Scaled Training Features ---")
                try:
                    desc_stats_scl = X_train_scl.describe().transpose()
                    logging.info(f"[{group_name_iter}] Descriptive statistics of X_train_scaled:\n{desc_stats_scl.to_string()}")
                    if X_train_scl.isnull().values.any():
                        logging.warning(f"[{group_name_iter}] X_train_scaled CONTAINS NaNs!")
                    if np.isinf(X_train_scl.values).any(): # Check on .values for DataFrame
                        logging.warning(f"[{group_name_iter}] X_train_scaled CONTAINS Infs!")
                except Exception as e_desc_scl:
                    logging.error(f"[{group_name_iter}] Error analyzing scaled features: {e_desc_scl}", exc_info=True)

        except Exception as e_split_scale:
            logging.error(f"[{group_name_iter}] Error during data splitting or scaling: {e_split_scale}", exc_info=True)
            # Continue to next hypothesis if it doesn't depend on split/scaled data

        # HYPOTHESIS 4: Collinearity in Original (unscaled) Training Features
        # This uses X_train_grp (unscaled)
        if not X_train_grp.empty:
            logging.info(f"--- HYPOTHESIS 4: Collinearity in Original Training Features ---")
            try:
                if X_train_grp.shape[1] > 1:
                    corr_mat_orig = X_train_grp.corr().abs()
                    upper_tri = corr_mat_orig.where(np.triu(np.ones(corr_mat_orig.shape), k=1).astype(bool))
                    max_c = upper_tri.max().max()
                    avg_c = upper_tri.stack().mean() # Get mean of non-NaN values in upper triangle
                    high_c_pairs_count = (upper_tri > 0.95).sum().sum()
                    logging.info(f"[{group_name_iter}] Correlation summary for X_train_grp (original):")
                    logging.info(f"    Max correlation (off-diagonal): {max_c:.4f}")
                    logging.info(f"    Avg correlation (off-diagonal, upper triangle): {avg_c:.4f}")
                    logging.info(f"    Number of pairs with abs_correlation > 0.95: {high_c_pairs_count}")
                    if high_c_pairs_count > 5 or max_c > 0.98:
                        logging.info(f"    Top 5 highest correlated pairs (abs values):")
                        sorted_corrs_display = upper_tri.stack().sort_values(ascending=False)
                        logging.info(f"\n{sorted_corrs_display.head(5).to_string()}")
                else:
                    logging.info(f"[{group_name_iter}] Skipping correlation analysis for X_train_grp as it has {X_train_grp.shape[1]} features.")
            except Exception as e_corr:
                logging.error(f"[{group_name_iter}] Error analyzing feature collinearity: {e_corr}", exc_info=True)
        else:
            logging.info(f"[{group_name_iter}] Skipping collinearity check as X_train_grp is empty.")


        logging.info(f"Finished investigation for Group: {group_name_iter}")
        print(f"Finished investigation for Group: {group_name_iter}")

    logging.info("SVM performance investigation script finished.")
    print("Investigation script finished.")

if __name__ == "__main__":
    investigate()