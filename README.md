# 🔧 Industrial Predictive Maintenance System (AI4I 2020)

### 📊 Condition Monitoring & Failure Diagnosis Dashboard
**Developed by:** Shivam Maurya

This project provides an end-to-end AI solution for predicting machinery failures and diagnosing their specific mechanical causes. By utilizing the **AI4I 2020 Predictive Maintenance Dataset**, the system translates complex sensor telemetry into actionable maintenance decisions.

🌐 **Live App:** [https://industrialfailurediagnosissystem-gjwz9qgrtdybke4oejrwpc.streamlit.app/](https://industrialfailurediagnosissystem-gjwz9qgrtdybke4oejrwpc.streamlit.app/)

---

## 🛡️ Project Overview & Methodology

### 1. Data Science Foundation
* **Model:** An **XGBoost Multi-Output Classifier** was implemented to simultaneously predict the probability of failure and 5 specific failure modes.
* **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to address the extreme class imbalance (3.4% failure rate) in the training data.
* **Optimal Decision Logic:** Instead of a standard 0.5 threshold, this system uses an **Optimal Threshold of 0.09 (9%)** derived from a **Precision-Recall Curve** (AUC: 0.54) to maximize the recall of critical failures.



### 2. Physics-Based Feature Engineering
To improve diagnostic accuracy, the following mechanical indicators were engineered:
* **Power Factor ($Torque \times RPM$):** Primary indicator for **Power Failures (PWF)**.
* **Temperature Delta ($Process - Air$):** Essential for detecting **Heat Dissipation Failures (HDF)**.
* **Overstrain Index ($Tool Wear \times Torque$):** Key predictor for **Overstrain Failures (OSF)**.

---

## 🚀 Key Features

* **Interactive Gauges:** Real-time Plotly visualizations for RPM, Torque, and Process Temperature.
* **Diagnostic Breakdown:** Provides a confidence-based analysis (e.g., "TWF: 52.60%") with specific recommended maintenance actions.
* **Condition Monitoring Trends:** Simulated time-series graphs showing realistic fluctuations in sensor data over time.
* **Safe Range Reference:** Built-in documentation for standard operating parameters to assist non-expert users.



---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* Libraries: `streamlit`, `xgboost`, `scikit-learn`, `plotly`, `joblib`, `pandas`, `numpy`

### Local Setup
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ishivamm/industrial_failure_diagnosis_system.git]
   cd industrial_failure_diagnosis_system
   pip install -r requirements.txt
   streamlit run app.py
