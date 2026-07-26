"""
BCAPM Configuration Module
Defines paths, hyperparameters, column mappings, and project settings.
"""
import sys
from pathlib import Path

# Base Directory & sys.path setup
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Directory Paths
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
PROJECT_DIR = BASE_DIR / "project"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure all directories exist
for path in [DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR, REPORTS_DIR, MODELS_DIR, PROJECT_DIR, NOTEBOOKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
ZIP_PATH = BASE_DIR / "Final.zip"
MASTER_V1_PATH = DATA_INTERIM_DIR / "BCAPM_Master_V1.csv"
MASTER_V2_PATH = DATA_INTERIM_DIR / "BCAPM_Master_V2.csv"
CLEAN_PATH = DATA_PROCESSED_DIR / "BCAPM_Clean.csv"
ENGINEERED_PATH = DATA_PROCESSED_DIR / "BCAPM_Engineered.csv"
PREPROCESSED_PATH = DATA_PROCESSED_DIR / "BCAPM_Preprocessed.csv"
SELECTED_FEATURES_PATH = REPORTS_DIR / "SelectedFeatures.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# Hyperparameters & Settings
RANDOM_SEED = 42
CURRENT_YEAR = 2026
TARGET_COL = "target_success"
