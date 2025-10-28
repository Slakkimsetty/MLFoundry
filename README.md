<h1 align="center">⚙️ MLFoundry</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg" />
  <img src="https://img.shields.io/badge/Library-scikit--learn-F7931E.svg" />
  <img src="https://img.shields.io/badge/ML-XGBoost-00C7B7.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

> **MLFoundry** is a self-evolving AutoML platform that automatically trains, compares, and explains multiple machine learning models.  
> It identifies the best-performing model, visualizes performance, and enables easy retraining.

---

## 🚀 Features
- 🔁 Automatically trains multiple ML models (Logistic Regression, XGBoost, Random Forest, etc.)
- 📊 Compares models by accuracy and F1-score (model leaderboard)
- 🧠 Shows feature importance for explainability
- 💾 Saves and version-controls the best model (`best_model.pkl`)
- 🧩 Streamlit dashboard for visualization, prediction, and retraining
- 🧮 CSV upload support for batch predictions
- 🧱 Designed for extensibility — plug in new models or datasets easily

---

## 🧠 Tech Stack
| Category | Tools |
|-----------|--------|
| **Languages** | Python |
| **ML / AI** | scikit-learn, XGBoost, pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Automation** | joblib, datetime, os |
| **UI / Dashboard** | Streamlit |
| **Version Control** | Git + GitHub |
| **(Optional)** | MLflow / Evidently AI for tracking |

---

## 🔮 Future Improvements
- Add feature importance visualization using SHAP
- Integrate MLflow for experiment tracking
- Include model drift monitoring (Evidently AI)
- Deploy live dashboard on Streamlit Cloud

---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/MLFoundry.git
cd MLFoundry
