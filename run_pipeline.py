"""
BCAPM Master Pipeline Execution Script
Orchestrates end-to-end data ingestion, merging, cleaning, feature engineering,
preprocessing, feature selection, EDA, model training, and analogy search.

Usage:
    python run_pipeline.py
"""
import sys
import time
import importlib
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    MASTER_V1_PATH, MASTER_V2_PATH, CLEAN_PATH, ENGINEERED_PATH, PREPROCESSED_PATH
)
from src.utils import get_logger, ensure_data_extracted

logger = get_logger("Pipeline_Orchestrator")

def main():
    logger.info("==========================================================")
    logger.info("  BCAPM DATA ENGINEERING & ML PIPELINE STARTED")
    logger.info("==========================================================")
    
    start_time = time.time()
    ensure_data_extracted()
    
    # Dynamically import modular pipeline stages
    c01 = importlib.import_module("src.01_collect_data")
    m02 = importlib.import_module("src.02_merge_v1")
    m03 = importlib.import_module("src.03_merge_v2")
    c04 = importlib.import_module("src.04_clean_data")
    f05 = importlib.import_module("src.05_feature_engineering")
    p06 = importlib.import_module("src.06_preprocessing")
    s07 = importlib.import_module("src.07_feature_selection")
    e08 = importlib.import_module("src.08_eda")
    t09 = importlib.import_module("src.09_train_models")
    s10 = importlib.import_module("src.10_similarity_engine")
    
    # Stage 1: Ingestion Audit
    logger.info("--> Running 01_collect_data.py...")
    c01.collect_and_audit_data()
    
    # Stage 2: Merge V1
    logger.info("--> Running 02_merge_v1.py...")
    df_v1 = m02.merge_v1()
    
    # Stage 3: Merge V2
    logger.info("--> Running 03_merge_v2.py...")
    df_v2 = m03.merge_v2()
    
    # Stage 4: Clean Data
    logger.info("--> Running 04_clean_data.py...")
    df_clean = c04.clean_data()
    
    # Stage 5: Feature Engineering
    logger.info("--> Running 05_feature_engineering.py...")
    df_eng = f05.engineer_features()
    
    # Stage 6: Preprocessing
    logger.info("--> Running 06_preprocessing.py...")
    df_prep = p06.preprocess_data()
    
    # Stage 7: Feature Selection
    logger.info("--> Running 07_feature_selection.py...")
    s07.select_features()
    
    # Stage 8: EDA
    logger.info("--> Running 08_eda.py...")
    e08.run_eda()
    
    # Stage 9: Model Training
    logger.info("--> Running 09_train_models.py...")
    t09.train_and_evaluate_models()
    
    # Stage 10: Similarity Engine Query Test
    logger.info("--> Testing 10_similarity_engine.py...")
    engine = s10.BCAPMSimilarityEngine()
    test_res = engine.find_analogs_by_name("Dropbox", top_k=3)
    logger.info(f"Similarity Engine Test Output: {test_res['Explanation']}")
    
    elapsed = time.time() - start_time
    
    logger.info("==========================================================")
    logger.info(f"  ALL PIPELINE STAGES EXECUTED SUCCESSFULLY IN {elapsed:.2f}s")
    logger.info("==========================================================")
    logger.info(f"  Artifact V1:   {MASTER_V1_PATH}")
    logger.info(f"  Artifact V2:   {MASTER_V2_PATH}")
    logger.info(f"  Artifact Clean:{CLEAN_PATH}")
    logger.info(f"  Artifact Prep: {PREPROCESSED_PATH}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
