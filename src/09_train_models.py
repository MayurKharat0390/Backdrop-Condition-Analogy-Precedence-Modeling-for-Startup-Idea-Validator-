"""
09_train_models.py: Multi-Model Machine Learning Training & Evaluation.

Purpose:
Train 5 ML algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting / XGBoost
- Artificial Neural Network (ANN via MLPClassifier)

Evaluate performance across metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

Export model comparison table to reports/model_comparison.csv and save trained models to models/.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import importlib
preprocess_data = importlib.import_module('src.06_preprocessing').preprocess_data

from src.config import PREPROCESSED_PATH, MODEL_COMPARISON_PATH, MODELS_DIR, TARGET_COL, RANDOM_SEED
from src.utils import get_logger, validate_df, load_csv_file

logger = get_logger("09_Train_Models")

def train_and_evaluate_models(input_path: str = None, models_dir: str = None) -> pd.DataFrame:
    """Train multiple classification models, evaluate metrics, and save model artifacts."""
    logger.info("Executing Stage 8: Model Training & Evaluation...")
    
    if input_path is None:
        input_path = PREPROCESSED_PATH
    else:
        input_path = Path(input_path)
        
    if not input_path.exists():
        logger.info(f"{input_path} not found. Running Stage 5 first...")
        df_prep = preprocess_data()
    else:
        df_prep = load_csv_file(input_path)
        
    if models_dir is None:
        models_dir = MODELS_DIR
    else:
        models_dir = Path(models_dir)
        
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loaded dataset: {len(df_prep)} rows. Target: '{TARGET_COL}'. Saving models to {models_dir}")
    
    ignore_cols = ['clean_name', 'name', TARGET_COL]
    feature_cols = [c for c in df_prep.columns if c not in ignore_cols]
    
    X = df_prep[feature_cols].fillna(0.0)
    y = df_prep[TARGET_COL].astype(int)
    
    # Train-test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")
    
    # Define Model Portfolio
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
        "ANN (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_SEED)
    }
    
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, eval_metric='logloss')
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)
    
    results = []
    
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        logger.info(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": auc
        })
        
        # Save trained model artifact
        save_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(model, models_dir / save_name)
        
    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    
    logger.info("==========================================================")
    logger.info("  MODEL EVALUATION COMPARISON SUMMARY")
    logger.info("==========================================================")
    logger.info(f"\n{results_df.to_string(index=False)}")
    logger.info(f"Saved evaluation metrics comparison table to: {MODEL_COMPARISON_PATH}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ML models.")
    parser.add_argument("--input", type=str, default=None, help="Path to BCAPM_Preprocessed.csv")
    parser.add_argument("--models", type=str, default=None, help="Path to models output directory")
    args = parser.parse_args()
    
    train_and_evaluate_models(input_path=args.input, models_dir=args.models)
