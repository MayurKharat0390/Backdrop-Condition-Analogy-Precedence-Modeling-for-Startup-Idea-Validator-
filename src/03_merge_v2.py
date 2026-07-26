"""
03_merge_v2.py: Macroeconomic Backdrop & Environmental Conditioning Integration.

Purpose:
- Load BCAPM_Master_V1.
- Ingest World Bank Internet Penetration Indicator (API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv).
- Ingest World Bank Country Metadata & Income Groups.
- Ingest National Economy Policy & HDI Indicators (D10/Final - Copy.csv).
- Ingest Macro Startup Failure Rates (D10/Startup_failure.csv) & Industry Baselines (D10/startup_indus.csv).
- Merge backdrop features onto startup entity records using country and founding year resolution.
- Save to data/interim/BCAPM_Master_V2.csv and export reports/merge_v2_report.txt.
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
from src.config import DATA_RAW_DIR, MASTER_V1_PATH, MASTER_V2_PATH, REPORTS_DIR
from src.utils import get_logger, validate_df, ensure_data_extracted, load_csv_file
import importlib
merge_v1 = importlib.import_module('src.02_merge_v1').merge_v1

logger = get_logger("03_Merge_V2")

def merge_v2(input_v1_path: str = None, output_path: str = None) -> pd.DataFrame:
    """Merge macro-economic backdrop indicators into BCAPM_Master_V1."""
    ensure_data_extracted()
    logger.info("Executing Stage 2: Building BCAPM_Master_V2...")
    
    # 1. Load Stage 1 Output
    if input_v1_path is None:
        input_v1_path = MASTER_V1_PATH
    else:
        input_v1_path = Path(input_v1_path)
        
    if not input_v1_path.exists():
        logger.info(f"{input_v1_path} not found. Running Stage 1 first...")
        df_v1 = merge_v1()
    else:
        df_v1 = load_csv_file(input_v1_path)
        
    logger.info(f"Loaded BCAPM_Master_V1: {len(df_v1)} records.")
    
    # 2. Ingest World Bank Internet Penetration Indicator
    wb_net_file = DATA_RAW_DIR / "API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv"
    logger.info(f"Loading World Bank Internet Penetration: {wb_net_file}")
    df_wb_net = load_csv_file(wb_net_file, skiprows=4)
    
    year_cols = [c for c in df_wb_net.columns if c.isdigit()]
    df_wb_melted = df_wb_net.melt(
        id_vars=['Country Code'], 
        value_vars=year_cols, 
        var_name='Year', 
        value_name='internet_penetration_pct'
    )
    df_wb_melted['Country Code'] = df_wb_melted['Country Code'].astype(str).str.strip()
    df_wb_melted['Year'] = pd.to_numeric(df_wb_melted['Year'], errors='coerce')
    df_wb_melted['internet_penetration_pct'] = pd.to_numeric(df_wb_melted['internet_penetration_pct'], errors='coerce')
    
    # 3. Ingest World Bank Country Metadata
    wb_meta_file = DATA_RAW_DIR / "Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv"
    logger.info(f"Loading World Bank Country Metadata: {wb_meta_file}")
    df_wb_meta = load_csv_file(wb_meta_file)
    df_wb_meta_sub = df_wb_meta[['Country Code', 'Region', 'IncomeGroup']].drop_duplicates()
    df_wb_meta_sub['Country Code'] = df_wb_meta_sub['Country Code'].astype(str).str.strip()
    
    # 4. Ingest Economy Policy & HDI Indicators (D10/Final - Copy.csv)
    econ_file = DATA_RAW_DIR / "D10" / "Final - Copy.csv"
    logger.info(f"Loading National Economic Indicators: {econ_file}")
    df_econ = load_csv_file(econ_file)
    econ_cols = {
        'Code': 'Country Code',
        'HDI': 'country_hdi',
        'Financing for entrepreneurs': 'entrepreneurial_financing_index',
        'Governmental support and policies': 'government_support_index',
        'Taxes and bureaucracy': 'tax_bureaucracy_index'
    }
    df_econ_sub = df_econ[[c for c in econ_cols.keys() if c in df_econ.columns]].rename(columns=econ_cols)
    df_econ_sub['Country Code'] = df_econ_sub['Country Code'].astype(str).str.strip()
    df_econ_sub = df_econ_sub.drop_duplicates(subset=['Country Code'])
    
    for c in ['country_hdi', 'entrepreneurial_financing_index', 'government_support_index', 'tax_bureaucracy_index']:
        if c in df_econ_sub.columns:
            df_econ_sub[c] = pd.to_numeric(df_econ_sub[c], errors='coerce')

    # 5. Ingest Annual Macro Failure Rate Backdrop (D10/Startup_failure.csv)
    fail_file = DATA_RAW_DIR / "D10" / "Startup_failure.csv"
    logger.info(f"Loading Macro Startup Failure Rates: {fail_file}")
    df_fail = load_csv_file(fail_file)
    df_fail['Year'] = pd.to_numeric(df_fail['Year'], errors='coerce')
    df_fail['macro_failure_rate_at_founding'] = pd.to_numeric(df_fail['Failure'].astype(str).str.replace('%', ''), errors='coerce') / 100.0
    df_fail_sub = df_fail[['Year', 'macro_failure_rate_at_founding']].drop_duplicates()
    
    # Map country code (default USA for US records)
    df_v1['country_code_3'] = df_v1['country_code'].replace({'USA': 'USA', 'US': 'USA'}).fillna('USA').astype(str).str.strip()
    df_v1['founded_year_num'] = pd.to_numeric(df_v1['founded_year'], errors='coerce')
    
    # Merge Backdrop Features
    logger.info("Merging World Bank Internet Penetration & Country Metadata...")
    master_v2 = pd.merge(
        df_v1,
        df_wb_melted,
        left_on=['country_code_3', 'founded_year_num'],
        right_on=['Country Code', 'Year'],
        how='left'
    ).rename(columns={'internet_penetration_pct': 'internet_penetration_at_founding'})
    
    master_v2 = pd.merge(master_v2, df_wb_meta_sub, left_on='country_code_3', right_on='Country Code', how='left')
    master_v2 = pd.merge(master_v2, df_econ_sub, left_on='country_code_3', right_on='Country Code', how='left')
    master_v2 = pd.merge(master_v2, df_fail_sub, left_on='founded_year_num', right_on='Year', how='left')
    
    drop_temp = [c for c in master_v2.columns if c.startswith('Country Code') or c.startswith('Year_')]
    drop_temp.extend(['country_code_3', 'founded_year_num'])
    master_v2 = master_v2.drop(columns=[c for c in drop_temp if c in master_v2.columns])
    
    if output_path is None:
        output_path = MASTER_V2_PATH
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_v2.to_csv(output_path, index=False)
    
    # Save text report
    report_text = f"""==================================================
BCAPM STAGE 2 MERGE REPORT
==================================================
Total Input V1 Records: {len(df_v1):,}
Total Output V2 Records: {len(master_v2):,}
Total Features: {master_v2.shape[1]}
Internet Penetration Coverage: {master_v2['internet_penetration_at_founding'].notna().sum():,} / {len(master_v2):,}
Country HDI Coverage: {master_v2['country_hdi'].notna().sum():,} / {len(master_v2):,}
Macro Failure Rate Coverage: {master_v2['macro_failure_rate_at_founding'].notna().sum():,} / {len(master_v2):,}
==================================================
"""
    report_path = REPORTS_DIR / "merge_v2_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
        
    validate_df(master_v2, "BCAPM_Master_V2", check_target=True)
    logger.info(f"Successfully exported BCAPM_Master_V2 to: {output_path}")
    logger.info(f"Exported merge report to: {report_path}")
    return master_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge macroeconomic backdrop features into BCAPM_Master_V2.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Master_V1.csv")
    parser.add_argument("--output", type=str, default=None, help="Output path for BCAPM_Master_V2.csv")
    args = parser.parse_args()
    
    merge_v2(input_v1_path=args.input, output_path=args.output)
