"""
05_feature_engineering.py: Domain-Specific Feature Engineering.

Purpose:
Create engineered features:
- StartupAge
- FundingVelocity
- FundingPerYear
- FundingDensity
- FounderExperienceScore
- InvestorDiversityScore
- StartupMaturity
- BackdropScore
- MacroStartupClimateScore
- MarketPopularityScore

Save updated dataset to data/processed/BCAPM_Engineered.csv.
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
from src.config import CLEAN_PATH, ENGINEERED_PATH, CURRENT_YEAR, TARGET_COL
from src.utils import get_logger, validate_df, save_csv_file
import importlib
clean_data = importlib.import_module('src.04_clean_data').clean_data

logger = get_logger("05_Feature_Engineering")

def engineer_features(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Generate domain-specific financial, operational, founder, and macro backdrop features."""
    logger.info("Executing Stage 4: Feature Engineering...")
    
    if input_path is None:
        input_path = CLEAN_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 3 first...")
        df_clean = clean_data()
    else:
        df_clean = pd.read_csv(input_path, low_memory=False)
        
    logger.info(f"Loaded BCAPM_Clean: {len(df_clean)} rows.")
    df_eng = df_clean.copy()
    
    # 1. StartupAge
    df_eng['founded_year'] = pd.to_numeric(df_eng['founded_year'], errors='coerce').fillna(2010)
    df_eng['StartupAge'] = np.maximum(0, CURRENT_YEAR - df_eng['founded_year'])
    
    # 2. FundingVelocity ($ Raised / Rounds Count)
    rounds = df_eng['funding_rounds_count'].fillna(0)
    df_eng['FundingVelocity'] = df_eng['funding_total_usd'] / (rounds + 1.0)
    
    # 3. FundingPerYear ($ Raised / Age)
    df_eng['FundingPerYear'] = df_eng['funding_total_usd'] / (df_eng['StartupAge'] + 1.0)
    
    # 4. FundingDensity (Log funding per founder)
    founders = df_eng['founder_count'].fillna(1).replace(0, 1)
    df_eng['FundingDensity'] = np.log1p(np.maximum(0, df_eng['funding_total_usd'] / founders))
    
    # 5. FounderExperienceScore
    cofounders = df_eng['cax_cofounders'].fillna(1)
    top_corp = df_eng['worked_in_top_companies'].fillna(0)
    df_eng['FounderExperienceScore'] = cofounders * 0.4 + top_corp * 1.5 + df_eng['founder_count'].fillna(1) * 0.3
    
    # 6. InvestorDiversityScore
    repeat_inv = df_eng['repeat_investor_count'].fillna(0)
    df_eng['InvestorDiversityScore'] = repeat_inv * 1.2 + rounds * 0.8
    
    # 7. StartupMaturity (Categorical Bucket: Early <= 2 yrs, Growth 3-6 yrs, Mature > 6 yrs)
    def assign_maturity(age):
        if age <= 2:
            return 'Early'
        elif age <= 6:
            return 'Growth'
        else:
            return 'Mature'
    df_eng['StartupMaturity'] = df_eng['StartupAge'].apply(assign_maturity)
    
    # 8. BackdropScore (Country HDI + Entrepreneurial Financing - Tax Bureaucracy)
    hdi = df_eng['country_hdi'].fillna(0.7)
    financing = df_eng['entrepreneurial_financing_index'].fillna(5.0)
    tax = df_eng['tax_bureaucracy_index'].fillna(5.0)
    df_eng['BackdropScore'] = (hdi * 10.0) + financing - (tax * 0.5)
    
    # 9. MacroStartupClimateScore (Internet Penetration * (1 - Failure Rate))
    net_pct = df_eng['internet_penetration_at_founding'].fillna(50.0) / 100.0
    fail_rate = df_eng['macro_failure_rate_at_founding'].fillna(0.2)
    df_eng['MacroStartupClimateScore'] = net_pct * (1.0 - fail_rate)
    
    # 10. MarketPopularityScore (HN Sentiment * Public Engagement)
    sentiment = df_eng['hn_sentiment_score'].fillna(0.0)
    engagement = df_eng['hn_public_engagement'].fillna(0.0)
    df_eng['MarketPopularityScore'] = sentiment * np.log1p(np.maximum(0, engagement))
    
    if output_path is None:
        output_path = ENGINEERED_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv_file(df_eng, output_path, index=False)
    
    validate_df(df_eng, "BCAPM_Engineered", check_target=True)
    logger.info(f"Successfully exported BCAPM_Engineered to: {output_path}")
    return df_eng

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate domain-specific engineered features.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Clean.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for BCAPM_Engineered.csv")
    args = parser.parse_args()
    
    engineer_features(input_path=args.input, output_path=args.output)
