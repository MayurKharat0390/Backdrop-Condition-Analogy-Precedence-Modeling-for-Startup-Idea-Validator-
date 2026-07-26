"""
08_eda.py: Exploratory Data Analysis & Visualization Suite.

Purpose:
Generate and export plots to reports/:
- Distribution plots
- Correlation heatmap
- Missing values visualization
- Funding histogram
- Startup age histogram
- Target variable balance chart
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import CLEAN_PATH, ENGINEERED_PATH, REPORTS_DIR, TARGET_COL
from src.utils import get_logger, load_csv_file
import importlib
engineer_features = importlib.import_module('src.05_feature_engineering').engineer_features

logger = get_logger("08_EDA")

def run_eda(input_path: str = None, reports_dir: str = None):
    """Generate comprehensive EDA charts and save to reports directory."""
    logger.info("Executing Stage 7: Exploratory Data Analysis (EDA)...")
    
    if input_path is None:
        input_path = ENGINEERED_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 4 first...")
        df = engineer_features()
    else:
        df = load_csv_file(input_path)
        
    if reports_dir is None:
        reports_dir = REPORTS_DIR
    else:
        reports_dir = Path(reports_dir)
        
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loaded dataset: {len(df)} rows. Saving plots to {reports_dir}")
    
    # 1. Target Balance Chart
    plt.figure(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts()
    plt.bar(['Failed/Closed (0)', 'Operating/Acquired (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#e74c3c', '#2ecc71'])
    plt.title("BCAPM Target Variable Balance (Success vs Failure)")
    plt.ylabel("Number of Startups")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(reports_dir / "target_balance.png", dpi=300)
    plt.close()
    logger.info("Generated target_balance.png")
    
    # 2. Funding Histogram (Log Transformed)
    plt.figure(figsize=(7, 4))
    log_funding = np.log1p(df['funding_total_usd'].clip(lower=0))
    plt.hist(log_funding, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Total Funding USD (Log Scale)")
    plt.xlabel("Log(1 + Funding Total USD)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(reports_dir / "funding_histogram.png", dpi=300)
    plt.close()
    logger.info("Generated funding_histogram.png")
    
    # 3. Startup Age Histogram
    plt.figure(figsize=(7, 4))
    plt.hist(df['StartupAge'].dropna(), bins=25, color='#9b59b6', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Startup Age (Years)")
    plt.xlabel("Age in Years (as of 2026)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(reports_dir / "startup_age_histogram.png", dpi=300)
    plt.close()
    logger.info("Generated startup_age_histogram.png")
    
    # 4. Correlation Heatmap
    num_cols = ['funding_total_usd', 'funding_rounds_count', 'founder_count', 
                'hn_sentiment_score', 'internet_penetration_at_founding', 
                'country_hdi', 'StartupAge', 'BackdropScore', TARGET_COL]
    valid_num_cols = [c for c in num_cols if c in df.columns]
    
    plt.figure(figsize=(9, 7))
    corr = df[valid_num_cols].corr()
    cax = plt.matshow(corr, fignum=1, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(cax)
    plt.xticks(range(len(valid_num_cols)), valid_num_cols, rotation=45, ha='left', fontsize=8)
    plt.yticks(range(len(valid_num_cols)), valid_num_cols, fontsize=8)
    plt.title("Feature Correlation Matrix", pad=30)
    plt.tight_layout()
    plt.savefig(reports_dir / "correlation_heatmap.png", dpi=300)
    plt.close()
    logger.info("Generated correlation_heatmap.png")
    
    # 5. Distribution Plots Multi-panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(df['hn_sentiment_score'].dropna(), bins=20, color='#1abc9c', edgecolor='black')
    axes[0, 0].set_title("HN Sentiment Score")
    
    axes[0, 1].hist(df['internet_penetration_at_founding'].dropna(), bins=20, color='#f39c12', edgecolor='black')
    axes[0, 1].set_title("Internet Penetration at Founding (%)")
    
    axes[1, 0].hist(df['BackdropScore'].dropna(), bins=20, color='#d35400', edgecolor='black')
    axes[1, 0].set_title("Backdrop Score")
    
    axes[1, 1].hist(df['FounderExperienceScore'].dropna(), bins=20, color='#2c3e50', edgecolor='black')
    axes[1, 1].set_title("Founder Experience Score")
    
    plt.tight_layout()
    plt.savefig(reports_dir / "distribution_plots.png", dpi=300)
    plt.close()
    logger.info("Generated distribution_plots.png")
    
    # 6. Missing Values Visualization
    missing_series = df.isna().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(8, 4))
    if not missing_series.empty:
        missing_series.plot(kind='bar', color='#e74c3c')
        plt.title("Top Missing Values per Feature")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, "Zero Missing Values in Clean Dataset!", ha='center', va='center', fontsize=14)
        plt.title("Missing Values Status")
    plt.tight_layout()
    plt.savefig(reports_dir / "missing_values.png", dpi=300)
    plt.close()
    logger.info("Generated missing_values.png")
    
    logger.info(f"EDA Complete! All plots saved to {reports_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA and generate charts.")
    parser.add_argument("--input", type=str, default=None, help="Path to input dataset CSV.")
    parser.add_argument("--reports", type=str, default=None, help="Output directory for plots.")
    args = parser.parse_args()
    
    run_eda(input_path=args.input, reports_dir=args.reports)
