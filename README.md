# Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.9785-green) ![PR--AUC](https://img.shields.io/badge/PR--AUC-0.7974-brightgreen)

A machine learning system that detects fraudulent credit card transactions in real time, built by a fraud analyst with 2+ years investigating financial crime at DTB Bank Kenya.

---

## What it does

- Classifies transactions as fraudulent or legitimate in real time
- Shows fraud probability with LOW / MEDIUM / HIGH risk levels  
- Explains every decision using SHAP waterfall charts
- Handles extreme class imbalance (0.17% fraud rate) using SMOTE

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9785** |
| Precision-Recall AUC | **0.7974** |
| Fraud recall | **89%** |
| Training samples | 454,902 (after SMOTE) |

---

## Key insights from the data

- Fraud transactions average $122 vs $88 for legitimate ones
- Fraudsters avoid very large amounts — max fraud was $2,125 vs $25,691 legitimate
- Fraud is spread evenly across all time periods — time alone cannot detect it
- Features V14, V12, V10 and V4 are the strongest fraud signals

---

## Why this project is different

Most fraud detection projects are built from textbooks. This one was built by someone who has:
- Investigated real fraud and money laundering cases at a major Kenyan bank
- Used NetGuardian fraud prevention systems in production
- Built and enforced real fraud prevention policies and risk frameworks

The model reflects patterns that actually matter to fraud teams, not just academic benchmarks.

---

## Tech stack

- **Model:** XGBoost classifier with SMOTE for class imbalance
- **Explainability:** SHAP TreeExplainer for per-transaction explanations
- **App:** Streamlit with real-time prediction
- **Data:** Kaggle Credit Card Fraud dataset (284,807 transactions, 492 fraud cases)

---

## Run locally

```bash
git clone https://github.com/tonyspetea/fraud-detector.git
cd fraud-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Antony Mutwiri** — Data Scientist & Fraud Analyst  
[GitHub](https://github.com/tonyspetea) • Nairobi, Kenya  

> Open to remote data science roles and freelance projects in fraud detection, financial analytics, and machine learning.
