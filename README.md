# Customer Purchase Prediction - Empirical ML Classification Study

A full end-to-end machine learning pipeline that predicts which customers will respond to a marketing campaign, enabling a company to maximise campaign ROI through targeted outreach rather than broad-spectrum marketing.

---

## Problem Statement

A company's marketing campaign has a **15% response rate** - meaning 85% of contacted customers are non-respondents. Contacting every customer wastes budget and dilutes campaign effectiveness. The goal is to build a classifier that identifies likely responders in advance, so the campaign targets only those customers, maximising profit per contact.

---

## Project Structure

```
Customer-purchase-prediction/
├── 01_Report/
│   └── Customers purchase prediction.pdf   # Full written research report
├── 02_Code/
│   ├── 01_EDA.ipynb                         # Exploratory data analysis
│   ├── 02_Feature_engineering.ipynb         # Preprocessing & feature engineering
│   └── 03_Models.ipynb                      # Model benchmarking & selection
└── Data.xlsx                                # Raw customer dataset (2,240 records)
```

---

## Dataset

- **2,240 customer records** with 15 features
- **Class imbalance**: 15% positive (accepted campaign), 85% negative
- Features include demographic, behavioral, and purchase history attributes

---

## Pipeline Overview

### 1. Exploratory Data Analysis (`01_EDA.ipynb`)
- Distribution analysis across all features
- Response rate breakdown and class imbalance quantification
- Missing value detection and handling strategy

### 2. Feature Engineering (`02_Feature_engineering.ipynb`)
- **Outlier removal**: IQR, Standard Deviation, and DBSCAN methods benchmarked and compared
- **Stratified train/test split** preserving class proportions
- **SMOTE oversampling** to address 15% class imbalance
- Feature selection using K-Best scoring

### 3. Model Benchmarking (`03_Models.ipynb`)

Benchmarked **19+ classifiers** across 4 categories:

| Category | Models |
|---|---|
| Standard | Logistic Regression, Naive Bayes (3 variants), SVC, NuSVC, Random Forest, XGBoost, Gradient Boosting, SGD, Decision Tree, KNN, Ridge, LDA, QDA, Nearest Centroid, Passive Aggressive, Label Propagation |
| Ensemble | Bagging, Voting Classifier |
| Neural Networks | MLP, Keras Deep Learning |
| Hyperparameter Tuned | GridSearchCV on top candidates |

**Evaluation methodology:**
- StratifiedKFold cross-validation across all candidates
- ROC Curve and Precision-Recall Curve analysis (chosen for imbalanced dataset)
- Ensemble voting with GridSearchCV-tuned top models

---

## Key Innovation - Profit-Optimised Classification Threshold

Instead of using the standard 0.5 decision boundary, a **custom profit-optimising threshold function** was engineered:

- Evaluated classification cutoff empirically across CV folds
- Optimised threshold to maximise campaign revenue, not just accuracy
- Applied KS statistic generalisation test for threshold robustness
- Produced interpretable profit-per-threshold curves

This reflects real-world deployment thinking: the business objective is profit, not raw F1 score.

---

## Results

- Top model selected via GridSearchCV hyperparameter tuning
- Best F1 score achieved through ensemble voting of tuned classifiers
- Profit-optimised threshold outperforms default 0.5 boundary in campaign revenue simulation
- Model exported and saved for production deployment

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML pipeline, cross-validation, model benchmarking |
| XGBoost | Gradient boosted trees |
| Keras / TensorFlow | Deep learning (MLP, neural networks) |
| imbalanced-learn | SMOTE oversampling |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualisation |

---

## Key Takeaways

- Rigorous preprocessing (outlier detection, stratified splits, SMOTE) is essential before model selection
- No single algorithm dominates across all metrics - ensemble methods consistently outperform individual classifiers on imbalanced data
- Replacing the default classification threshold with a profit-optimised boundary materially improves business outcomes beyond what accuracy or F1 alone would suggest
