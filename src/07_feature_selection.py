"""
07_feature_selection.py: Feature Selection & Relevance Ranking.

Purpose:
- Implement Pearson Correlation analysis.
- Implement Mutual Information classification scoring.
- Apply Variance Threshold filtering.
- Compute Random Forest Gini Feature Importance.
- Generate and export ranked feature importance list to reports/SelectedFeatures.csv.
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
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from src.config import PREPROCESSED_PATH, SELECTED_FEATURES_PATH, TARGET_COL, RANDOM_SEED
from src.utils import get_logger, validate_df, load_csv_file, save_csv_file
import importlib
preprocess_data = importlib.import_module('src.06_preprocessing').preprocess_data

logger = get_logger("07_Feature_Selection")

def select_features(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Evaluate feature relevance using Pearson correlation, Mutual Info, Variance, and Random Forest."""
    logger.info("Executing Stage 6: Feature Selection & Ranking...")
    
    if input_path is None:
        input_path = PREPROCESSED_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 5 first...")
        df_prep = preprocess_data()
    else:
        df_prep = load_csv_file(input_path)
        
    logger.info(f"Loaded BCAPM_Preprocessed: {len(df_prep)} rows, {df_prep.shape[1]} cols.")
    
    # Drop non-feature identifier columns
    ignore_cols = ['clean_name', 'name', TARGET_COL]
    feature_cols = [c for c in df_prep.columns if c not in ignore_cols]
    
    X = df_prep[feature_cols].fillna(0.0)
    y = df_prep[TARGET_COL].astype(int)
    
    # 1. Variance Threshold Filtering
    logger.info("Applying Variance Threshold (threshold=0.01)...")
    selector_var = VarianceThreshold(threshold=0.01)
    selector_var.fit(X)
    variances = selector_var.variances_
    
    # 2. Pearson Correlation with Target
    logger.info("Computing Pearson Correlations...")
    correlations = X.apply(lambda col: col.corr(y)).abs()
    
    # 3. Mutual Information Classification
    logger.info("Computing Mutual Information Scores...")
    # Sample up to 10k rows for efficiency if dataset is large
    sample_n = min(10000, len(X))
    idx_sample = X.sample(sample_n, random_state=RANDOM_SEED).index
    mi_scores = mutual_info_classif(X.loc[idx_sample], y.loc[idx_sample], random_state=RANDOM_SEED)
    
    # 4. Random Forest Feature Importance
    logger.info("Computing Random Forest Feature Importances...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X.loc[idx_sample], y.loc[idx_sample])
    rf_importances = rf.feature_importances_
    
    # Build Combined Ranking Table
    ranking_df = pd.DataFrame({
        "Feature": feature_cols,
        "Variance": variances,
        "Pearson_Correlation": correlations.values,
        "Mutual_Information": mi_scores,
        "RF_Importance": rf_importances
    })
    
    # Calculate Composite Importance Score (Normalized Rank Sum)
    ranking_df['RF_Rank'] = ranking_df['RF_Importance'].rank(ascending=False)
    ranking_df['MI_Rank'] = ranking_df['Mutual_Information'].rank(ascending=False)
    ranking_df['Corr_Rank'] = ranking_df['Pearson_Correlation'].rank(ascending=False)
    
    ranking_df['Composite_Rank'] = (ranking_df['RF_Rank'] + ranking_df['MI_Rank'] + ranking_df['Corr_Rank']) / 3.0
    ranking_df = ranking_df.sort_values(by='Composite_Rank', ascending=True).reset_index(drop=True)
    
    if output_path is None:
        output_path = SELECTED_FEATURES_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv_file(ranking_df, output_path, index=False)
    
    logger.info(f"Feature Selection Complete. Saved top ranked features to: {output_path}")
    logger.info(f"Top 10 Features:\n{ranking_df[['Feature', 'RF_Importance', 'Mutual_Information', 'Composite_Rank']].head(10)}")
    return ranking_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform feature selection and rank importance.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Preprocessed.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for SelectedFeatures.csv")
    args = parser.parse_args()
    
    select_features(input_path=args.input, output_path=args.output)
