"""
run_economic_experiment.py: Master Experimental Benchmark for the BCAPM Venture Capital Economic Decision Layer.

Evaluates 5 Controlled Models on the EXACT SAME Untouched Test Set:
- Model A: Standard Conventional Classifier (Gradient Boosting / BCE)
- Model B: Class-Weighted Classifier (class_weight='balanced' / balanced sample weights)
- Model C: SMOTE Resampled Classifier (SMOTE on X_train only)
- Model D: Asymmetric Loss Objective (w1=1.0, w0=3.0)
- Model E: BCAPM Integrated Economic Decision Model (Calibrated Probabilities + EMV Layer)

Outputs:
- reports/vc_model_comparison.csv
- reports/vc_threshold_sweep.csv
- reports/vc_sensitivity_matrix.csv
- 10 Publication-Grade Visualizations in reports/
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    brier_score_loss, roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

from src.config import PREPROCESSED_PATH, REPORTS_DIR, TARGET_COL, RANDOM_SEED
from src.utils import get_logger, load_csv_file, save_csv_file
from src.economic_utility import calculate_emv, calculate_optimal_threshold, make_investment_decision
from src.asymmetric_loss import AsymmetricGradientBoostingClassifier, compute_sample_weights_for_asymmetric_loss
from src.calibration import evaluate_calibration, fit_calibrator
from src.portfolio_simulation import simulate_portfolio
from src.threshold_optimization import sweep_thresholds, run_sensitivity_analysis
from src.evaluation import evaluate_vc_model

logger = get_logger("VC_Economic_Experiment")


def run_experiment(
    investment_amount: float = 1.0,
    success_payoff: float = 20.0
):
    logger.info("=" * 70)
    logger.info("STARTING BCAPM VENTURE CAPITAL ECONOMIC DECISION EXPERIMENT")
    logger.info(f"Baseline Economic Scenario: Check Size = ${investment_amount:.2f}M | Success Payoff = ${success_payoff:.2f}M")
    p_star = calculate_optimal_threshold(investment_amount, success_payoff)
    logger.info(f"Bayesian Breakeven Threshold p* = I / V = {p_star:.4f} ({p_star*100:.2f}%)")
    logger.info("=" * 70)

    # 1. Load Preprocessed Data
    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PREPROCESSED_PATH} not found. Please run preprocessing first.")
        
    df = load_csv_file(PREPROCESSED_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows.")

    ignore_cols = ['clean_name', 'name', TARGET_COL]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols].fillna(0.0).values
    y = df[TARGET_COL].astype(int).values
    
    # 2. Train / Val / Test Split (60% Train, 20% Val for Calibration, 20% Untouched Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    logger.info(f"Data Partitions: Train={len(X_train)} | Val (Calibration)={len(X_val)} | Test (Evaluation)={len(X_test)}")
    logger.info(f"Test Set Class Distribution: Success (1) = {np.sum(y_test==1)} ({np.mean(y_test)*100:.2f}%) | Failure (0) = {np.sum(y_test==0)} ({100-np.mean(y_test)*100:.2f}%)")

    # =========================================================================
    # 3. Model Training on Identical Training Partition
    # =========================================================================
    models = {}
    
    # Model A: Standard Conventional Classifier (HistGradientBoosting / BCE)
    logger.info("Training Model A: Standard Classifier (Conventional BCE)...")
    clf_a = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
    clf_a.fit(X_train, y_train)
    models["Model A: Standard (BCE)"] = clf_a

    # Model B: Class-Weighted Classifier
    logger.info("Training Model B: Class-Weighted Classifier (Balanced Loss)...")
    w_pos = len(y_train) / (2.0 * np.sum(y_train == 1))
    w_neg = len(y_train) / (2.0 * np.sum(y_train == 0))
    sample_weights_b = np.where(y_train == 1, w_pos, w_neg)
    clf_b = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
    clf_b.fit(X_train, y_train, sample_weight=sample_weights_b)
    models["Model B: Class-Weighted"] = clf_b

    # Model C: SMOTE Resampled Classifier (Applied Strictly to X_train)
    logger.info("Training Model C: SMOTE Resampled Classifier...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    clf_c = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED)
    clf_c.fit(X_train_smote, y_train_smote)
    models["Model C: SMOTE Resampled"] = clf_c

    # Model D: Asymmetric Loss Objective (w1=1.0, w0=3.0)
    logger.info("Training Model D: Asymmetric Economic Loss Objective...")
    clf_d = AsymmetricGradientBoostingClassifier(max_iter=100, max_depth=6, w1=1.0, w0=3.0, random_state=RANDOM_SEED)
    clf_d.fit(X_train, y_train)
    models["Model D: Asymmetric Loss"] = clf_d

    # Model E: BCAPM Integrated Economic Decision Model (Platt Calibrated on Val Set)
    logger.info("Calibrating Model E: BCAPM Economic Decision Model (Platt Calibrated)...")
    clf_e = fit_calibrator(clf_a, X_val, y_val, method="sigmoid")
    models["Model E: BCAPM Economic Model"] = clf_e

    # =========================================================================
    # 4. Comprehensive Multi-Layer Evaluation on Untouched Test Set
    # =========================================================================
    logger.info("=" * 70)
    logger.info("EVALUATING ALL 5 MODELS ON UNTOUCHED TEST SET")
    logger.info("=" * 70)

    comparison_records = []
    test_probs = {}

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        test_probs[name] = probs
        
        # Model A uses standard 0.50 threshold; others evaluated at the economic threshold p* = 0.05
        thresh = 0.50 if "Model A" in name else p_star
        eval_result = evaluate_vc_model(
            model_name=name,
            y_true=y_test,
            y_prob=probs,
            threshold=thresh,
            investment_amount=investment_amount,
            success_payoff=success_payoff
        )
        comparison_records.append(eval_result)
        logger.info(
            f"[{name}] Thresh: {thresh:.2f} | Acc: {eval_result['Accuracy']:.4f} | "
            f"PR-AUC: {eval_result['PR_AUC']:.4f} | ROI: {eval_result['Portfolio_ROI']:.2f}x | "
            f"Net Return: ${eval_result['Portfolio_Net_Return_M']:.1f}M | "
            f"Opportunity Loss: ${eval_result['Opportunity_Loss_M']:.1f}M"
        )

    df_comparison = pd.DataFrame(comparison_records)
    comp_path = REPORTS_DIR / "vc_model_comparison.csv"
    save_csv_file(df_comparison, comp_path, index=False)
    logger.info(f"Exported model comparison table to: {comp_path}")

    # =========================================================================
    # 5. Multi-Threshold Sweep for Model E (BCAPM Economic Model)
    # =========================================================================
    logger.info("Executing Multi-Threshold Sweep for BCAPM Model...")
    p_e = test_probs["Model E: BCAPM Economic Model"]
    df_sweep = sweep_thresholds(y_test, p_e, investment_amount=investment_amount, success_payoff=success_payoff)
    sweep_path = REPORTS_DIR / "vc_threshold_sweep.csv"
    save_csv_file(df_sweep, sweep_path, index=False)
    logger.info(f"Exported threshold sweep to: {sweep_path}")

    # =========================================================================
    # 6. Economic Sensitivity Matrix Analysis
    # =========================================================================
    logger.info("Running 2D Economic Sensitivity Analysis...")
    df_sens = run_sensitivity_analysis(y_test, p_e)
    sens_path = REPORTS_DIR / "vc_sensitivity_matrix.csv"
    save_csv_file(df_sens, sens_path, index=False)
    logger.info(f"Exported sensitivity matrix to: {sens_path}")

    # =========================================================================
    # 7. Generate 10 Research-Grade Publication Figures
    # =========================================================================
    logger.info("Generating 10 Research-Grade Visualizations...")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Confusion Matrix (with Financial Annotations)
    decisions_e = (p_e >= p_star).astype(int)
    cm = confusion_matrix(y_test, decisions_e)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Predicted Reject (0)', 'Predicted Invest (1)'],
                yticklabels=['Actual Failure (0)', 'Actual Success (1)'])
    ax.set_title(f"Venture Capital Confusion Matrix\n(Threshold p* = {p_star:.2f})", fontsize=13, fontweight='bold')
    # Annotate financial impact
    fp = cm[0, 1]
    fn = cm[1, 0]
    ax.text(1.5, 0.25, f"Capital Loss:\n-${fp * investment_amount:.1f}M", ha='center', va='center', color='darkred', fontweight='bold')
    ax.text(0.5, 1.25, f"Opportunity Cost:\n${fn * success_payoff:.1f}M", ha='center', va='center', color='darkorange', fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_confusion_matrix.png", dpi=300)
    plt.close()

    # Plot 2: ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=13, fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_roc_curve.png", dpi=300)
    plt.close()

    # Plot 3: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
    ax.set_title("Precision-Recall Curve Benchmark", fontsize=13, fontweight='bold')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_pr_curve.png", dpi=300)
    plt.close()

    # Plot 4: Calibration Curve (Raw vs Calibrated)
    fig, ax = plt.subplots(figsize=(8, 6))
    p_raw = test_probs["Model A: Standard (BCE)"]
    eval_raw = evaluate_calibration(y_test, p_raw)
    eval_cal = evaluate_calibration(y_test, p_e)
    ax.plot(eval_raw["prob_pred"], eval_raw["prob_true"], "s-", label=f"Model A: Raw (Brier = {eval_raw['brier_score']:.4f})", lw=2)
    ax.plot(eval_cal["prob_pred"], eval_cal["prob_true"], "o-", label=f"Model E: Calibrated (Brier = {eval_cal['brier_score']:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Probability Calibration Diagram (Reliability Curve)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Predicted Success Probability")
    ax.set_ylabel("Empirical Fraction of Successes")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_calibration_curve.png", dpi=300)
    plt.close()

    # Plot 5: Threshold vs Portfolio ROI
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df_sweep["Threshold"], df_sweep["Portfolio_ROI"], marker='o', color='#2b5c8f', lw=2.5)
    ax.axvline(p_star, color='crimson', linestyle='--', label=f'Optimal Hurdle Rate p* = {p_star:.2f}')
    ax.set_title("Investment Threshold vs. Portfolio Return on Investment (ROI)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Decision Threshold (p)")
    ax.set_ylabel("Portfolio ROI Multiple")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_threshold_vs_roi.png", dpi=300)
    plt.close()

    # Plot 6: Threshold vs Total Expected Monetary Value (EMV)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df_sweep["Threshold"], df_sweep["Total_EMV_M"], marker='s', color='#2e7d32', lw=2.5)
    ax.axvline(p_star, color='crimson', linestyle='--', label=f'Theoretical Hurdle Rate p* = {p_star:.2f}')
    ax.set_title("Investment Threshold vs. Total Expected Monetary Value (EMV)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Decision Threshold (p)")
    ax.set_ylabel("Total Portfolio EMV ($ Millions)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_threshold_vs_emv.png", dpi=300)
    plt.close()

    # Plot 7: Threshold vs Deal Flow Selection Rate & Recall
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df_sweep["Threshold"], df_sweep["Selection_Rate"], marker='^', color='#e65100', lw=2, label='Selection Rate (% Deals Funded)')
    ax.plot(df_sweep["Threshold"], df_sweep["Recall"], marker='v', color='#1565c0', lw=2, label='Recall (Winners Captured)')
    ax.set_title("Tradeoff: Deal Flow Selection Rate vs. Winner Capture Recall", fontsize=13, fontweight='bold')
    ax.set_xlabel("Decision Threshold (p)")
    ax.set_ylabel("Rate / Fraction")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_threshold_vs_recall_selection.png", dpi=300)
    plt.close()

    # Plot 8: Capital Deployed vs Portfolio Net Return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(df_sweep["Capital_Deployed_M"], df_sweep["Portfolio_Net_Return_M"], 
                         c=df_sweep["Threshold"], cmap='viridis', s=120, edgecolors='black', zorder=3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Decision Threshold")
    ax.set_title("Capital Deployed vs. Portfolio Net Return Efficiency", fontsize=13, fontweight='bold')
    ax.set_xlabel("Total Capital Deployed ($ Millions)")
    ax.set_ylabel("Portfolio Net Return ($ Millions)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_capital_deployed_vs_net_return.png", dpi=300)
    plt.close()

    # Plot 9: Opportunity Loss vs Direct Capital Loss across Thresholds
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df_sweep["Threshold"], df_sweep["Opportunity_Loss_M"], marker='d', color='darkorange', lw=2.5, label='Opportunity Loss (FN * $20M)')
    ax.plot(df_sweep["Threshold"], df_sweep["Capital_Loss_M"], marker='x', color='darkred', lw=2.5, label='Capital Loss (FP * $1M)')
    ax.set_title("Asymmetric Cost Balance: Opportunity Loss vs. Capital Loss", fontsize=13, fontweight='bold')
    ax.set_xlabel("Decision Threshold (p)")
    ax.set_ylabel("Loss Magnitude ($ Millions)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_opportunity_loss_vs_threshold.png", dpi=300)
    plt.close()

    # Plot 10: Portfolio Return Distribution
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sim_e = simulate_portfolio(y_test, decisions_e, investment_amount, success_payoff)
    deal_returns = sim_e["Deal_Returns_Array"]
    sns.histplot(deal_returns, bins=20, kde=False, color='#3f51b5', ax=ax)
    ax.axvline(0, color='black', linestyle='--', lw=1.5)
    ax.set_title(f"Simulated Portfolio Deal-by-Deal Return Distribution\n(Total Net Return: ${sim_e['Portfolio_Net_Return_M']:.1f}M)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Net Return per Deal ($ Millions)")
    ax.set_ylabel("Number of Funded Investments")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "vc_portfolio_return_distribution.png", dpi=300)
    plt.close()

    logger.info("=" * 70)
    logger.info("ALL 10 PUBLICATION-GRADE PLOTS SUCCESSFULLY EXPORTED TO reports/")
    logger.info("=" * 70)

    return df_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BCAPM VC Economic Decision Benchmark.")
    parser.add_argument("--investment", type=float, default=1.0, help="Check size in $M (default: 1.0)")
    parser.add_argument("--payoff", type=float, default=20.0, help="Exit payoff in $M (default: 20.0)")
    args = parser.parse_args()
    
    run_experiment(investment_amount=args.investment, success_payoff=args.payoff)
