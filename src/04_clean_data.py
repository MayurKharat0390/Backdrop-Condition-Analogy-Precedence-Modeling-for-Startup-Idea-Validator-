"""
04_clean_data.py: Data Hygiene, Imputation & Type Normalization.

Purpose:
- Handle missing values (sector median imputation + global default fallback).
- Remove duplicates.
- Convert dates into clean numeric years.
- Convert numeric columns safely.
- Standardize country names and company names.
- Save to data/processed/BCAPM_Clean.csv.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import MASTER_V2_PATH, CLEAN_PATH, TARGET_COL
from src.utils import get_logger, validate_df, save_csv_file
import importlib
merge_v2 = importlib.import_module('src.03_merge_v2').merge_v2

logger = get_logger("04_Clean_Data")

def clean_data(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Perform data cleaning, missing value imputation, and type normalization."""
    logger.info("Executing Stage 3: Cleaning dataset for BCAPM_Clean...")
    
    if input_path is None:
        input_path = MASTER_V2_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 2 first...")
        df_v2 = merge_v2()
    else:
        df_v2 = pd.read_csv(input_path, low_memory=False)
        
    logger.info(f"Loaded BCAPM_Master_V2: {len(df_v2)} rows.")
    
    # 1. Strict Deduplication by clean_name
    df_clean = df_v2.drop_duplicates(subset=['clean_name']).copy()
    logger.info(f"Deduplicated by clean_name: {len(df_clean)} rows.")
    
    # 2. Target Variable Cleaning (Drop rows missing target label)
    df_clean = df_clean.dropna(subset=[TARGET_COL]).copy()
    df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(int)
    logger.info(f"Valid target label records: {len(df_clean)} rows.")
    
    # 3. Clean Dates & Convert Numeric Columns
    df_clean['founded_year'] = pd.to_numeric(df_clean['founded_year'], errors='coerce').fillna(2010).astype(int)
    
    # 4. Standardize Country Codes
    df_clean['country_code'] = df_clean['country_code'].replace({'US': 'USA', 'USA': 'USA'}).fillna('USA').astype(str).str.strip().str.upper()
    
    # 5. Missing Value Imputation for Numeric Features
    num_cols = [
        'funding_total_usd', 'funding_rounds_count', 'founder_count',
        'female_founder_ratio', 'hn_sentiment_score', 'hn_public_engagement',
        'cax_cofounders', 'team_senior_leadership_size', 'repeat_investor_count',
        'internet_penetration_at_founding', 'country_hdi',
        'entrepreneurial_financing_index', 'government_support_index',
        'tax_bureaucracy_index', 'macro_failure_rate_at_founding'
    ]
    
    for col in num_cols:
        if col in df_clean.columns:
            sector_medians = df_clean.groupby('market_category')[col].transform('median')
            global_median = df_clean[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            df_clean[col] = df_clean[col].fillna(sector_medians).fillna(global_median)
            
    # 6. Categorical Variable Imputation
    cat_cols = ['market_category', 'country_code', 'state_code', 'city', 'Region', 'IncomeGroup']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str)
            
    # Binary Indicators Imputation
    bin_cols = ['worked_in_top_companies', 'is_ml_based', 'b2c_b2b_venture']
    for col in bin_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int)
            
    if output_path is None:
        output_path = CLEAN_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv_file(df_clean, output_path, index=False)
    
    validate_df(df_clean, "BCAPM_Clean", check_target=True)
    logger.info(f"Successfully exported BCAPM_Clean to: {output_path}")
    return df_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and impute dataset into BCAPM_Clean.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Master_V2.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for BCAPM_Clean.csv")
    args = parser.parse_args()
    
    clean_data(input_path=args.input, output_path=args.output)
