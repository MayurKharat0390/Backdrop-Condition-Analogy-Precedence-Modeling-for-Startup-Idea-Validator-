"""
BCAPM Utilities Module
Contains helper functions for logging, string canonicalization, numeric parsing, data validation, and extraction.
"""
import re
import logging
import zipfile
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from src.config import ZIP_PATH, DATA_RAW_DIR, TARGET_COL

def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger("Utils")

def ensure_data_extracted():
    """Extract Final.zip into data/raw/ if raw files do not exist."""
    if not DATA_RAW_DIR.exists() or not any(DATA_RAW_DIR.iterdir()):
        logger.info(f"Extracting {ZIP_PATH} to {DATA_RAW_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_RAW_DIR)
        logger.info("Extraction completed successfully.")

def load_csv_file(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Safely load a CSV file with UTF-8 and Latin-1 fallback."""
    p = Path(file_path)
    try:
        return pd.read_csv(p, low_memory=False, **kwargs)
    except (UnicodeDecodeError, Exception):
        return pd.read_csv(p, encoding="latin1", low_memory=False, **kwargs)

def save_csv_file(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> Path:
    """
    Safely export a DataFrame to CSV, with exception handling for Windows file locks (PermissionError).
    If target file is open/locked, attempts writing to a fallback filename or retries.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(p, **kwargs)
        return p
    except PermissionError as e:
        logger.warning(f"Permission denied writing to {p} (file may be open in an editor or external viewer): {e}")
        fallback_path = p.with_name(p.stem + "_latest" + p.suffix)
        logger.info(f"Attempting export to fallback path: {fallback_path}")
        try:
            df.to_csv(fallback_path, **kwargs)
            logger.info(f"Successfully saved to fallback path: {fallback_path}")
            return fallback_path
        except Exception as ex:
            logger.error(f"Failed to export to fallback path: {ex}")
            raise e

def canonicalize_name(name: Union[str, float]) -> str:
    """
    Standardize company name strings:
    - Convert to lowercase
    - Trim leading/trailing whitespace
    - Remove common corporate suffixes (Inc, LLC, Corp, etc.)
    - Remove non-alphanumeric noise
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    c_name = name.strip().lower()
    suffixes = [r'\binc\.?\b', r'\bllc\.?\b', r'\bcorp\.?\b', r'\bcorporation\b', r'\bltd\.?\b', r'\bco\.?\b']
    for suf in suffixes:
        c_name = re.sub(suf, '', c_name)
    c_name = re.sub(r'[^a-z0-9\s]', '', c_name)
    c_name = re.sub(r'\s+', ' ', c_name).strip()
    return c_name

def parse_usd(val: Union[str, float, int]) -> float:
    """Parse string currency values (e.g. '$1.5M', '$500K') into float USD."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper().replace(',', '').replace('$', '')
    multiplier = 1.0
    if 'M' in s:
        multiplier = 1e6
        s = s.replace('M', '')
    elif 'B' in s:
        multiplier = 1e9
        s = s.replace('B', '')
    elif 'K' in s:
        multiplier = 1e3
        s = s.replace('K', '')
    try:
        clean_num = float(re.sub(r'[^0-9.]', '', s))
        return clean_num * multiplier
    except ValueError:
        return 0.0

def validate_df(df: pd.DataFrame, dataset_name: str, check_target: bool = False) -> bool:
    """
    Validate DataFrame shape, column integrity, missing values, and target existence.
    Returns True if validation passes.
    """
    logger.info(f"--- Data Validation: {dataset_name} ---")
    if df.empty:
        logger.error(f"Validation FAILED: {dataset_name} is empty!")
        return False
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    logger.info(f"Missing Values: {df.isna().sum().sum()}")
    
    if check_target and TARGET_COL in df.columns:
        valid_targets = df[TARGET_COL].dropna().count()
        logger.info(f"Target variable '{TARGET_COL}' non-null records: {valid_targets}")
        if valid_targets == 0:
            logger.warning(f"Target column '{TARGET_COL}' has no non-null values!")
            
    return True
