# Backdrop-Conditioned Analogy Precedent Modeling (BCAPM) for Startup Success Prediction

An end-to-end production Machine Learning pipeline for predicting startup success through **Backdrop-Conditioned Analogy Precedent Modeling (BCAPM)**. This research project integrates multi-source micro-level startup firmographics, founder human capital metrics, Hacker News community sentiment, and World Bank macroeconomic backdrop indicators.

---

## 🚨 Problem Statement

* **High Failure Rate**: Over 90% of technology startups fail within their first 5 years, leading to billions of dollars in misallocated venture capital and entrepreneurial failure.
* **Flaws of Naive Category Matching**: Traditional startup assessment tools evaluate new ventures based solely on superficial industry tags (e.g. comparing a new SaaS tool to 1990s legacy software), missing cross-industry structural growth playbooks.
* **Neglect of Macroeconomic Backdrop**: Existing models fail to account for external macro environmental conditions at founding—such as national internet penetration rates, human development index (HDI), tax/bureaucracy friction, and macro failure rates.
* **Lack of Explainability**: Standard machine learning classifiers act as black boxes, giving a prediction score without identifying *which historical startups succeeded or failed under identical conditions*.

---

## 🛡️ Proposed Solution: BCAPM Architecture

**Backdrop-Conditioned Analogy Precedent Modeling (BCAPM)** addresses these challenges by combining machine learning success classification with explainable precedent retrieval:

1. **Multi-Source Data Fusion**: Harmonizes micro-level firmographics (Crunchbase, Y Combinator, Hacker News, CAX) with World Bank macroeconomic backdrop indicators across 65,930 historical startups.
2. **Domain-Specific Feature Engineering**: Constructs composite indices (`BackdropScore`, `MacroStartupClimateScore`, `FundingDensity`, `MarketPopularityScore`) capturing environmental readiness and traction velocity.
3. **Hybrid Ensemble Feature Selection**: Combines Variance Threshold filtering, Pearson Correlation, Mutual Information classification, and Random Forest Gini Impurity reduction into an unbiased **Composite Rank Score**.
4. **Multi-Model ML Benchmarking**: Evaluates 5 algorithms (Logistic Regression, Decision Tree, Random Forest, ANN, and Gradient Boosting), achieving **90.55% Accuracy** and **0.7034 ROC-AUC**.
5. **Analogy Precedent Engine**: Utilizes 70-dimensional Vector Cosine Similarity to find the Top-5 historical startup precedents, outputting explainable precedent-conditioned success probabilities.

---

## 📊 Model Evaluation Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **90.55%** | **90.66%** | **99.87%** | **0.9504** | **0.7034** |
| **Random Forest** | **90.62%** | **90.62%** | **100.00%** | **0.9508** | **0.6927** |
| **Decision Tree** | 90.50% | 90.63% | 99.83% | 0.9501 | 0.6757 |
| **ANN (Neural Network)** | 90.19% | 90.67% | 99.40% | 0.9483 | 0.6669 |
| **Logistic Regression** | 90.62% | 90.62% | 100.00% | 0.9508 | 0.6395 |

---

## 🔍 Hybrid Feature Selection Methodology

Rather than relying on a single feature selection filter (which can introduce algorithm bias), `src/07_feature_selection.py` implements a 4-stage **Hybrid Ensemble Framework**:

1. **Variance Threshold Filtering**: Removes quasi-constant features ($\text{Variance} < 0.01$).
2. **Pearson Correlation ($r$)**: Measures linear relationships with target success.
3. **Mutual Information (`mutual_info_classif`)**: Captures non-linear entropy dependencies ($I(X; Y) = H(X) - H(X|Y)$).
4. **Random Forest Gini Impurity**: Evaluates tree-split impurity reduction across decision nodes.

**Composite Rank Score**: $\text{Composite Rank} = \frac{\text{RF\_Rank} + \text{MI\_Rank} + \text{Corr\_Rank}}{3.0}$

---

## 📁 Project Structure

```
BCAMP/
├── DATASET_ANALYSIS.md          # Comprehensive catalog & audit of all 37 raw CSV datasets
├── PIPELINE_DESIGN.md           # In-depth 4-stage data pipeline architecture
├── Final.zip                    # Source raw dataset zip archive
├── run_pipeline.py              # Top-level CLI orchestrator running stages 1-10
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore settings
├── README.md                    # Project documentation
│
├── notebooks/                   # Jupyter notebooks for interactive analysis
├── reports/                     # Output figures, heatmaps, and evaluation tables
│   ├── dataset_collection_summary.csv
│   ├── merge_v2_report.txt
│   ├── SelectedFeatures.csv
│   ├── model_comparison.csv
│   └── *.png                    # EDA charts (distribution, heatmap, histograms)
│
├── models/                      # Serialized trained ML model artifacts (.joblib)
├── data/                        # Tiered data storage directory
│   ├── raw/                     # Extracted raw CSV files from Final.zip
│   ├── interim/                 # Staged master outputs (BCAPM_Master_V1, BCAPM_Master_V2)
│   └── processed/               # Cleaned & ML-ready tensors (BCAPM_Clean, BCAPM_Preprocessed)
│
└── src/                         # Modular production-grade Python package
    ├── config.py                # System paths, constants, and hyperparameters
    ├── utils.py                 # String canonicalization, currency parsing & validation
    ├── 01_collect_data.py       # Ingestion audit and schema report generation
    ├── 02_merge_v1.py           # Entity resolution & micro-level startup merge
    ├── 03_merge_v2.py           # Macroeconomic backdrop & World Bank indicator merge
    ├── 04_clean_data.py         # Missing value imputation & type normalization
    ├── 05_feature_engineering.py # Financial ratios, founder scores & climate indices
    ├── 06_preprocessing.py      # One-hot encoding & continuous feature scaling
    ├── 07_feature_selection.py  # Pearson, Mutual Info & Random Forest feature ranking
    ├── 08_eda.py                # Exploratory data visualization suite
    ├── 09_train_models.py       # Training Logistic Regression, DT, RF, Gradient Boosting & ANN
    ├── 10_similarity_engine.py  # BCAPM Cosine Similarity Analogy Precedent Engine
    └── similarity_engine.py    # Clean Python package import wrapper
```

---

## 🚀 Quick Start & Execution

### 1. Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Complete Data & ML Pipeline
Execute the full 10-stage pipeline:
```bash
python run_pipeline.py
```

### 3. Query the BCAPM Precedent Engine (CLI)

```bash
# Query historical startup in dataset (e.g. Airbnb, Dropbox, Uber)
python src/10_similarity_engine.py --company "Airbnb" --top_k 5
```

---

## 💻 Python API Usage

To query historical precedent analogs in Python scripts or Jupyter Notebooks:

```python
from src.similarity_engine import BCAPMSimilarityEngine

engine = BCAPMSimilarityEngine()
results = engine.find_analogs_by_name("Airbnb", top_k=5)

print("Predicted Success Probability:", results["Precedent_Success_Probability"])
for analog in results["Top_K_Analogs"]:
    print(f"- {analog['Company_Name']} (Similarity: {analog['Similarity_Score']:.4f})")
```

---

## 📜 Verification & Outputs

- **Dataset Audit Report**: [DATASET_ANALYSIS.md](file:///d:/PROJECTS/BCAPM/BCAMP/DATASET_ANALYSIS.md)
- **Pipeline Architecture Specs**: [PIPELINE_DESIGN.md](file:///d:/PROJECTS/BCAPM/BCAMP/PIPELINE_DESIGN.md)
- **Feature Selection Ranks**: `reports/SelectedFeatures.csv`
- **Model Evaluation Metrics**: `reports/model_comparison.csv`
- **EDA Visualizations**: `reports/*.png`
- **Trained Model Artifacts**: `models/*.joblib`
