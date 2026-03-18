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

## 🖥️ Application Preview

Below are screenshots showcasing different sections of the application:
![web-screenshot-22-02-2026 (3)](https://github.com/user-attachments/assets/65da309b-76ab-4620-bccf-2dba60e76c3e)
![web-screenshot-22-02-2026 (4)](https://github.com/user-attachments/assets/a95e7353-bd4b-49b3-b6da-29847156814e)
![web-screenshot-22-02-2026 (5)](https://github.com/user-attachments/assets/f726bcd4-1058-4c5e-ad60-801bf80c3f79)
![web-screenshot-22-02-2026 (6)](https://github.com/user-attachments/assets/cf89a689-7d6a-417b-982a-f9a2369707a7)
![web-screenshot-22-02-2026 (10)](https://github.com/user-attachments/assets/c2c7fc14-f9ab-4a79-926f-3a5fc3980423)

More sceernshots are in screenshot_app folder


---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* Libraries: `streamlit`, `xgboost`, `scikit-learn`, `plotly`, `joblib`, `pandas`, `numpy`

### Local Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ishivamm/industrial_failure_diagnosis_system.git
   cd industrial_failure_diagnosis_system
   pip install -r requirements.txt
   streamlit run app.py
