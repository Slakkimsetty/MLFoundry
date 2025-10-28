# ⚙️ MLFoundry

> **MLFoundry** is a self-evolving AutoML platform that automatically trains, evaluates, and explains multiple machine learning models.  
> It identifies the best-performing model, tracks its versions, and provides interactive dashboards for exploration and retraining.

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

## ⚙️ Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/MLFoundry.git
cd MLFoundry
