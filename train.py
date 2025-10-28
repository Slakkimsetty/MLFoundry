# ============================================
# train.py — AutoML Intelligence Platform
# ============================================
# Trains multiple ML models automatically,
# compares performance, and saves the best one.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

print("🚀 Starting AutoML training...")

# 1️⃣ Load data
data = pd.read_csv("data.csv")
print("✅ Data loaded successfully! Shape:", data.shape)

# 2️⃣ Identify target column (for Bank Marketing, it is usually 'deposit')
target_column = "deposit"  # change if your dataset has a different target column

if target_column not in data.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found! Available columns: {list(data.columns)}")

# Separate features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Convert target to numeric if it has Yes/No or categorical values
if y.dtype == 'object':
    y = y.map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

# Convert categorical columns to dummy variables
X = pd.get_dummies(X)

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Scale numeric features for models that need it
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Define all models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# 6️⃣ Train and evaluate all models
results = []

print("\n⚙️ Training models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    results.append((name, acc, f1))
    print(f"✅ {name:20s} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

# 7️⃣ Find and save best model
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1_Score"])
best_model_name = results_df.sort_values(by="F1_Score", ascending=False).iloc[0, 0]
best_model = models[best_model_name]

joblib.dump(best_model, "best_model.pkl")
results_df.to_csv("model_comparison.csv", index=False)

print("\n🏆 Best Model:", best_model_name)
print("📈 Model comparison saved as model_comparison.csv")
print("💾 Saved best model as best_model.pkl")
