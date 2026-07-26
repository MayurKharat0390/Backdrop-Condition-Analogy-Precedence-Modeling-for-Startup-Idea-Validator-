"""
01_collect_data.py: Raw Dataset Collection & Ingestion Audit.

Purpose:
- Ingest all CSV files extracted from Final.zip.
- Inspect schemas, column data types, row counts, duplicates, missing value totals.
- Generate and export a summary report CSV to reports/dataset_collection_summary.csv.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import glob
import argparse
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_DIR, REPORTS_DIR
from src.utils import get_logger, ensure_data_extracted, validate_df

logger = get_logger("01_Collect_Data")

def collect_and_audit_data(output_path: str = None) -> pd.DataFrame:
    """Load, audit, and generate schema report for all raw CSV files."""
    ensure_data_extracted()
    logger.info("Starting raw data collection and audit...")
    
    csv_files = sorted(glob.glob(os.path.join(str(DATA_RAW_DIR), "**", "*.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} CSV datasets in raw data directory.")
    
    summary_records = []
    
    for filepath in csv_files:
        rel_path = os.path.relpath(filepath, str(DATA_RAW_DIR)).replace("\\", "/")
        filename = os.path.basename(filepath)
        
        try:
            # World Bank main indicator dataset skiprows=4
            if filename.startswith("API_IT.NET.USER.ZS_DS2_en_csv_v2"):
                df = pd.read_csv(filepath, skiprows=4)
            else:
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
                    
            rows, cols = df.shape
            dups = int(df.duplicated().sum())
            missing_total = int(df.isna().sum().sum())
            sample_cols = ", ".join(list(df.columns[:5]))
            
            logger.info(f"Loaded {rel_path} | Rows: {rows:,} | Cols: {cols} | Dups: {dups:,} | Missing: {missing_total:,}")
            
            summary_records.append({
                "Dataset_Path": rel_path,
                "Filename": filename,
                "Row_Count": rows,
                "Column_Count": cols,
                "Duplicate_Rows": dups,
                "Missing_Values_Total": missing_total,
                "Sample_Columns": sample_cols
            })
        except Exception as e:
            logger.error(f"Failed to read {rel_path}: {e}")
            summary_records.append({
                "Dataset_Path": rel_path,
                "Filename": filename,
                "Row_Count": 0,
                "Column_Count": 0,
                "Duplicate_Rows": 0,
                "Missing_Values_Total": 0,
                "Sample_Columns": f"ERROR: {str(e)}"
            })
            
    summary_df = pd.DataFrame(summary_records)
    
    if output_path is None:
        output_path = REPORTS_DIR / "dataset_collection_summary.csv"
    else:
        output_path = Path(output_path)
        
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved dataset collection summary report to: {output_path}")
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and audit all raw datasets.")
    parser.add_argument("--output", type=str, default=None, help="Path to save summary report CSV.")
    args = parser.parse_args()
    
    collect_and_audit_data(output_path=args.output)
