"""
dataset_hierarchy.py: 3-Tier Outcome Hierarchy & Zero-Leakage Temporal Partitioner.

Outcome Formulations:
1. Outcome 1 (Survival): Operating + Acquired + IPO (1) vs. Closed (0).
   - N = 65,930 | ~90.6% Success vs 9.4% Failure (Reflects raw database survival).
2. Outcome 2 (Liquidity Exit): Acquired + IPO (1) vs. Closed (0).
   - N = 13,333 | 53.2% Success vs 46.8% Failure (NATURALLY BALANCED investor liquidity).
3. Outcome 3 (VC Outlier Success): Acquired/IPO with Funding >= $10M (1) vs. Closed/Low-exit (0).
   - N = 13,333 | 24.7% Success vs 75.3% Failure (Reflects power-law venture economics).

Temporal Partitioning:
- Train: founded_year <= 2010 (~50% of dated startups).
- Val (Calibration/Hurdle Tuning): founded_year in [2011, 2012] (~25%).
- Locked Out-of-Time Test: founded_year >= 2013 (~25%).
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd

from src.config import PREPROCESSED_PATH, CLEAN_PATH, DATA_RAW_DIR, TARGET_COL, RANDOM_SEED
from src.utils import get_logger, canonicalize_name, load_csv_file

logger = get_logger("Dataset_Hierarchy")


def load_hierarchy_dataframe() -> pd.DataFrame:
    """
    Assemble master preprocessed feature tensor with raw status and temporal metadata.
    """
    # 1. Load 100% numerical preprocessed features
    df_prep = load_csv_file(PREPROCESSED_PATH)
    
    # 2. Load Clean data for founded_year and funding_total_usd
    df_clean = pd.read_csv(CLEAN_PATH, usecols=['clean_name', 'name', 'founded_year', 'funding_total_usd'])
    
    # 3. Load raw status from companies.csv
    raw_comp_file = DATA_RAW_DIR / "D1" / "companies.csv"
    df_raw = pd.read_csv(raw_comp_file, usecols=['name', 'status'])
    df_raw['clean_name'] = df_raw['name'].apply(canonicalize_name)
    df_raw = df_raw.drop_duplicates(subset=['clean_name'])
    
    # Merge status onto clean metadata
    meta = pd.merge(df_clean, df_raw[['clean_name', 'status']], on='clean_name', how='left')
    meta['status'] = meta['status'].fillna('operating').str.lower()
    
    # Ensure index alignment
    df_combined = df_prep.copy()
    df_combined['raw_status'] = meta['status'].values
    df_combined['founded_year'] = meta['founded_year'].values
    df_combined['funding_total_usd'] = meta['funding_total_usd'].values
    df_combined['company_name'] = meta['name'].values
    
    logger.info(f"Loaded master hierarchy dataset: {len(df_combined)} rows.")
    return df_combined


def get_outcome_dataset(
    df: Optional[pd.DataFrame] = None,
    outcome_tier: str = "outcome_2_liquidity"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract feature matrix X, target vector y, and metadata for a specific outcome tier.
    
    Parameters:
        df: Master hierarchy dataframe (loaded if None).
        outcome_tier:
            - 'outcome_1_survival': Operating + Acquired + IPO vs Closed
            - 'outcome_2_liquidity': Acquired + IPO vs Closed (53:47 naturally balanced)
            - 'outcome_3_vc_outlier': Acquired/IPO >= $10M vs Closed/Low-exit (25:75)
            
    Returns:
        X: 2D feature matrix (numerical).
        y: 1D binary target array (0 or 1).
        sub_df: Filtered dataframe containing metadata and features.
    """
    if df is None:
        df = load_hierarchy_dataframe()
        
    sub = df.copy()
    
    if outcome_tier == "outcome_1_survival":
        # All startups: 1 = operating/acquired/ipo, 0 = closed
        sub['target'] = np.where(sub['raw_status'].isin(['operating', 'acquired', 'ipo']), 1, 0)
    elif outcome_tier == "outcome_2_liquidity":
        # Realized exit vs total write-off (Naturally balanced benchmark 53% : 47%)
        sub = sub[sub['raw_status'].isin(['acquired', 'ipo', 'closed'])].copy()
        sub['target'] = np.where(sub['raw_status'].isin(['acquired', 'ipo']), 1, 0)
    elif outcome_tier in ["outcome_3_vc_outlier", "outcome_3_well_funded_exit"]:
        # Well-Funded Liquidity Exit: Meaningful capital scale (cumulative funding >= $10M) + exit
        # Note on Research Integrity: Crunchbase acquisitions.csv has 73.6% missing acquisition prices,
        # and IPO market caps are omitted from the snapshot. Thus, this tier measures well-capitalized
        # liquidity events rather than verified transaction valuation.
        sub = sub[sub['raw_status'].isin(['acquired', 'ipo', 'closed'])].copy()
        is_high_exit = sub['raw_status'].isin(['acquired', 'ipo']) & (sub['funding_total_usd'] >= 10_000_000.0)
        sub['target'] = np.where(is_high_exit, 1, 0)
    else:
        raise ValueError(f"Unknown outcome_tier '{outcome_tier}'. Choose from outcome_1_survival, outcome_2_liquidity, outcome_3_well_funded_exit (or outcome_3_vc_outlier).")
        
    ignore_cols = ['clean_name', 'name', 'company_name', 'raw_status', 'founded_year', 'funding_total_usd', 'target', TARGET_COL]
    feature_cols = [c for c in sub.columns if c not in ignore_cols]
    
    X = sub[feature_cols].fillna(0.0).values
    y = sub['target'].astype(int).values
    
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_pct = float(pos_count / len(y) * 100) if len(y) > 0 else 0.0
    
    logger.info(
        f"[{outcome_tier}] Filtered rows: {len(sub)} | "
        f"Success (1): {pos_count} ({pos_pct:.2f}%) | "
        f"Failure (0): {neg_count} ({100-pos_pct:.2f}%)"
    )
    
    return X, y, sub


def get_temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    sub_df: pd.DataFrame,
    train_end_year: int = 2010,
    val_end_year: int = 2012
) -> Dict[str, np.ndarray]:
    """
    Perform zero-leakage temporal partitioning based on founding year.
    
    Partitions:
        Train: founded_year <= train_end_year (e.g. <= 2010)
        Val: train_end_year < founded_year <= val_end_year (e.g. 2011 - 2012)
        Test: founded_year > val_end_year (e.g. >= 2013)
    """
    years = sub_df['founded_year'].values
    
    train_mask = (years <= train_end_year)
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = (years > val_end_year)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(
        f"Temporal Partitioning -> "
        f"Train (<= {train_end_year}): {len(y_train)} rows ({np.mean(y_train)*100:.1f}% pos) | "
        f"Val ({train_end_year+1}-{val_end_year}): {len(y_val)} rows ({np.mean(y_val)*100:.1f}% pos) | "
        f"Test (>= {val_end_year+1}): {len(y_test)} rows ({np.mean(y_test)*100:.1f}% pos)"
    )
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "sub_test": sub_df[test_mask].copy()
    }
