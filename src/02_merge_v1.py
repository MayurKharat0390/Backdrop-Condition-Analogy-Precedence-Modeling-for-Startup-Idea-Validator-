"""
02_merge_v1.py: Micro-Level Startup Integration & Name Canonicalization.

Purpose:
- Load startup datasets (Crunchbase, YC, HN Sentiment, Founder profiles, CAX).
- Normalize company names (lowercase, trim spaces, strip suffixes, remove non-alphanumeric noise).
- Remove duplicate entity entries.
- Merge startup information into unified master record.
- Save to data/interim/BCAPM_Master_V1.csv.
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
from src.config import DATA_RAW_DIR, MASTER_V1_PATH, TARGET_COL
from src.utils import get_logger, canonicalize_name, parse_usd, validate_df, ensure_data_extracted, load_csv_file

logger = get_logger("02_Merge_V1")

def merge_v1(output_path: str = None) -> pd.DataFrame:
    """Perform entity resolution and merge raw micro-level startup datasets."""
    ensure_data_extracted()
    logger.info("Executing Stage 1: Building BCAPM_Master_V1...")
    
    # 1. Ingest Crunchbase Companies (D1/companies.csv)
    comp_file = DATA_RAW_DIR / "D1" / "companies.csv"
    logger.info(f"Loading companies: {comp_file}")
    df_comp = load_csv_file(comp_file)
    df_comp['clean_name'] = df_comp['name'].apply(canonicalize_name)
    df_comp = df_comp[df_comp['clean_name'] != ''].drop_duplicates(subset=['clean_name'])
    
    df_comp['funding_total_usd_clean'] = df_comp['funding_total_usd'].apply(parse_usd)
    df_comp['founded_year'] = pd.to_datetime(df_comp['founded_at'], errors='coerce').dt.year
    
    status_map = {'operating': 1, 'acquired': 1, 'closed': 0, 'ipo': 1}
    df_comp['target_success'] = df_comp['status'].str.lower().map(status_map)
    
    # 2. Ingest Crunchbase Funding Rounds (D1/rounds.csv)
    rounds_file = DATA_RAW_DIR / "D1" / "rounds.csv"
    logger.info(f"Loading funding rounds: {rounds_file}")
    df_rounds = load_csv_file(rounds_file)
    df_rounds['clean_name'] = df_rounds['company_name'].apply(canonicalize_name)
    rounds_agg = df_rounds.groupby('clean_name').agg(
        funding_rounds_count=('funding_round_permalink', 'count'),
        raised_amount_usd_sum=('raised_amount_usd', 'sum')
    ).reset_index()
    
    # 3. Ingest YC & Hacker News Datasets (D5/CleanStartupsFull3.csv & CompleteSet3.csv)
    yc_file = DATA_RAW_DIR / "D5" / "CleanStartupsFull3.csv"
    logger.info(f"Loading YC startups: {yc_file}")
    df_yc = load_csv_file(yc_file)
    df_yc['clean_name'] = df_yc['Company'].apply(canonicalize_name)
    df_yc['yc_total_funds'] = df_yc['TotalFunds'].apply(parse_usd)
    yc_fate_map = {'Operating': 1, 'Acquired': 1, 'Dead': 0}
    df_yc['yc_target_success'] = df_yc['Fate'].map(yc_fate_map)
    
    hn_file = DATA_RAW_DIR / "D5" / "CompleteSet3.csv"
    logger.info(f"Loading HN sentiment data: {hn_file}")
    df_hn = load_csv_file(hn_file)
    df_hn['clean_name'] = df_hn['Company'].apply(canonicalize_name)
    hn_agg = df_hn.groupby('clean_name').agg(
        hn_sentiment_score=('Sentiment', 'mean'),
        hn_title_points=('TitlePoints', 'max'),
        hn_comment_points=('TopCommentPoints', 'max')
    ).reset_index()
    hn_agg['hn_public_engagement'] = hn_agg['hn_title_points'].fillna(0) + hn_agg['hn_comment_points'].fillna(0)
    
    # 4. Ingest Founders Data (D5/Founders.csv & D2/founder_V0.3_founder.csv)
    founders_file = DATA_RAW_DIR / "D5" / "Founders.csv"
    logger.info(f"Loading YC founders: {founders_file}")
    df_f_yc = load_csv_file(founders_file)
    df_f_yc['clean_name'] = df_f_yc['Company'].apply(canonicalize_name)
    f_yc_agg = df_f_yc.groupby('clean_name').agg(
        founder_count=('Founder', 'count'),
        female_founder_count=('Gender', lambda g: (g.str.lower() == 'female').sum())
    ).reset_index()
    f_yc_agg['female_founder_ratio'] = f_yc_agg['female_founder_count'] / f_yc_agg['founder_count'].replace(0, 1)
    
    # 5. Ingest CAX Operational Datasets (D9/CAX_Startup_Data.csv)
    cax_file = DATA_RAW_DIR / "D9" / "CAX_Startup_Data.csv"
    logger.info(f"Loading CAX startup operational features: {cax_file}")
    df_cax = load_csv_file(cax_file)
    df_cax['clean_name'] = df_cax['Company_Name'].apply(canonicalize_name)
    cax_cols = [
        'clean_name', 'Number of Co-founders', 'Team size Senior leadership', 
        'Number of of repeat investors', 'Worked in top companies',
        'Machine Learning based business', 'B2C or B2B venture?'
    ]
    df_cax_sub = df_cax[cax_cols].drop_duplicates(subset=['clean_name']).rename(columns={
        'Number of Co-founders': 'cax_cofounders',
        'Team size Senior leadership': 'team_senior_leadership_size',
        'Number of of repeat investors': 'repeat_investor_count',
        'Worked in top companies': 'worked_in_top_companies',
        'Machine Learning based business': 'is_ml_based',
        'B2C or B2B venture?': 'b2c_b2b_venture'
    })
    
    # --- Execute Merges ---
    logger.info("Integrating micro-level components...")
    master = pd.merge(df_comp, rounds_agg, on='clean_name', how='left')
    master = pd.merge(master, df_yc[['clean_name', 'Market', 'yc_total_funds', 'yc_target_success', 'YCYear']], on='clean_name', how='left')
    master = pd.merge(master, hn_agg, on='clean_name', how='left')
    master = pd.merge(master, f_yc_agg, on='clean_name', how='left')
    master = pd.merge(master, df_cax_sub, on='clean_name', how='left')
    
    master['target_success'] = master['target_success'].fillna(master['yc_target_success'])
    master['market_category'] = master['category_list'].fillna(master['Market']).fillna('Software')
    master['funding_total_usd'] = master['funding_total_usd_clean'].replace(0, np.nan).fillna(master['yc_total_funds']).fillna(master['raised_amount_usd_sum']).fillna(0)
    
    v1_columns = [
        'clean_name', 'name', 'market_category', 'country_code', 'state_code', 'city',
        'founded_year', 'funding_total_usd', 'funding_rounds_count',
        'founder_count', 'female_founder_ratio', 'hn_sentiment_score', 'hn_public_engagement',
        'cax_cofounders', 'team_senior_leadership_size', 'repeat_investor_count',
        'worked_in_top_companies', 'is_ml_based', 'b2c_b2b_venture',
        'target_success'
    ]
    
    for col in v1_columns:
        if col not in master.columns:
            master[col] = np.nan
            
    master_v1 = master[v1_columns].copy()
    
    if output_path is None:
        output_path = MASTER_V1_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_v1.to_csv(output_path, index=False)
    
    validate_df(master_v1, "BCAPM_Master_V1", check_target=True)
    logger.info(f"Successfully exported BCAPM_Master_V1 to: {output_path}")
    return master_v1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw micro-level startup datasets into BCAPM_Master_V1.")
    parser.add_argument("--output", type=str, default=None, help="Output path for BCAPM_Master_V1.csv")
    args = parser.parse_args()
    
    merge_v1(output_path=args.output)
