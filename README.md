# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

An explainable machine learning platform and interactive Streamlit analytics application designed for retail banks to quantify churn risk, interpret behavioral attrition drivers, and launch proactive retention campaigns.

---

## 📌 Project Overview
- **Dataset:** 10,000 retail banking accounts across France, Germany, and Spain.
- **Baseline Churn Rate:** 20.37%.
- **Champion Architecture:** Gradient Boosted Decision Tree (ROC-AUC: 0.8694, Accuracy: 86.55%).
- **Frontline Decision Boundary:** Calibrated threshold $p \ge 0.30$ yielding 65.4% recall and a projected 372% campaign ROI.

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/bank-churn-intelligence-system.git
   cd bank-churn-intelligence-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Rebuild deliverables (PDFs):**
   ```bash
   python build_reports.py
   ```

4. **Launch the Streamlit web dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Core Application Modules
1. **Portfolio Risk Overview:** Real-time KPI summaries, risk tier distributions, and demographic cross-sections.
2. **Individual Risk Calculator:** Calculates customer churn probabilities and generates tailored intervention plans.
3. **What-If Sensitivity Simulator:** Simulates real-time risk reduction from cross-selling and digital activation.
4. **Model Benchmarks & Drivers:** Evaluates model performance and details GBDT feature importance.
5. **Customer Explorer & Batch Roster:** Filterable customer table with CSV export functionality.

---

## 📂 Deliverables Checklist
- `deliverables/Research_Paper.pdf` — Core research paper detailing methodology, model benchmarking, and business strategy.
- `deliverables/Executive_Summary.pdf` — Executive summary briefing tailored for bank directors and regulators.
- `app.py` — Live interactive Streamlit application.
- `data/` — Raw and scored customer datasets.
- `models/` — Serialized pipeline, benchmarks, and feature importance.
