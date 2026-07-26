"""
06_preprocessing.py: ML Preprocessing, Encoding & Continuous Feature Scaling.

Purpose:
- Apply One Hot Encoding and Label Encoding for categorical features.
- Scale continuous features using StandardScaler.
- Convert boolean indicators.
- Produce 100% numerical ML-ready dataset BCAPM_Preprocessed.csv and metadata mapping BCAPM_Metadata.csv.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import importlib
engineer_features = importlib.import_module('src.05_feature_engineering').engineer_features

from src.config import ENGINEERED_PATH, PREPROCESSED_PATH, DATA_PROCESSED_DIR, TARGET_COL
from src.utils import get_logger, validate_df, save_csv_file

logger = get_logger("06_Preprocessing")

def preprocess_data(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Preprocess dataset: encode categorical variables, scale numerical features, save tensor."""
    logger.info("Executing Stage 5: Preprocessing & Feature Scaling...")
    
    if input_path is None:
        input_path = ENGINEERED_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 4 first...")
        df_eng = engineer_features()
    else:
        df_eng = pd.read_csv(input_path, low_memory=False)
        
    logger.info(f"Loaded BCAPM_Engineered: {len(df_eng)} rows.")
    df_prep = df_eng.copy()
    
    # 1. Label Encoding for StartupMaturity
    le_mat = LabelEncoder()
    df_prep['StartupMaturity_Encoded'] = le_mat.fit_transform(df_prep['StartupMaturity'].astype(str))
    
    # 2. Boolean Indicator Conversion (0 / 1)
    bool_cols = ['worked_in_top_companies', 'is_ml_based', 'b2c_b2b_venture']
    for b in bool_cols:
        if b in df_prep.columns:
            df_prep[b] = df_prep[b].fillna(0).astype(int)
            
    # 3. Top-N Categorical Grouping for Market Categories
    top_markets = df_prep['market_category'].value_counts().head(30).index
    df_prep['market_category_clean'] = df_prep['market_category'].apply(lambda x: x if x in top_markets else 'Other')
    
    top_regions = df_prep['Region'].value_counts().head(15).index
    df_prep['region_clean'] = df_prep['Region'].apply(lambda x: x if x in top_regions else 'Other')
    
    # 4. Continuous Features to Scale
    continuous_cols = [
        'funding_total_usd', 'funding_rounds_count', 'founder_count',
        'female_founder_ratio', 'hn_sentiment_score', 'hn_public_engagement',
        'cax_cofounders', 'team_senior_leadership_size', 'repeat_investor_count',
        'internet_penetration_at_founding', 'country_hdi',
        'entrepreneurial_financing_index', 'government_support_index',
        'tax_bureaucracy_index', 'macro_failure_rate_at_founding',
        'StartupAge', 'FundingVelocity', 'FundingPerYear', 'FundingDensity',
        'FounderExperienceScore', 'InvestorDiversityScore', 'BackdropScore',
        'MacroStartupClimateScore', 'MarketPopularityScore'
    ]
    
    for c in continuous_cols:
        if c not in df_prep.columns:
            df_prep[c] = 0.0
            
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_prep[continuous_cols].fillna(0.0))
    df_scaled = pd.DataFrame(scaled_matrix, columns=[f"scaled_{c}" for c in continuous_cols], index=df_prep.index)
    
    # 5. One-Hot Encoding for Categoricals
    cat_nominal = ['market_category_clean', 'IncomeGroup', 'region_clean']
    df_ohe = pd.get_dummies(df_prep[cat_nominal], prefix=['market', 'income', 'region'], drop_first=True)
    
    # 6. Save Separate Metadata Mapping (clean_name, name)
    meta_df = df_prep[['clean_name', 'name']].copy()
    meta_path = DATA_PROCESSED_DIR / "BCAPM_Metadata.csv"
    meta_path = save_csv_file(meta_df, meta_path, index=False)
    
    # 7. Assemble 100% Numerical Preprocessed Dataset
    target_series = df_prep[TARGET_COL].astype(int)
    
    df_preprocessed = pd.concat([
        df_scaled,
        df_prep[['StartupMaturity_Encoded'] + bool_cols],
        df_ohe,
        target_series
    ], axis=1)
    
    if output_path is None:
        output_path = PREPROCESSED_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_csv_file(df_preprocessed, output_path, index=False)
    
    validate_df(df_preprocessed, "BCAPM_Preprocessed", check_target=True)
    logger.info(f"Successfully exported 100% numerical BCAPM_Preprocessed to: {output_path}")
    logger.info(f"Successfully exported metadata mapping to: {meta_path}")
    return df_preprocessed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and scale dataset into BCAPM_Preprocessed.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Engineered.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for BCAPM_Preprocessed.csv")
    args = parser.parse_args()
    
    preprocess_data(input_path=args.input, output_path=args.output)
