import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from joblib import dump
from data_preprocessing import preprocess

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../data/raw/train.csv")

y = df["Survived"]

# Preprocess data
X_processed, feature_columns = preprocess(
    df.drop(columns=["Survived"]),
    is_train=True
)

# Remove PassengerId from training features
X = X_processed.drop(columns=["PassengerId"])

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Define models
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, kernel="rbf"))
    ])
}

best_model = None
best_auc = 0

# -----------------------------
# Train, Evaluate, Cross-Validate
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    cv_auc = cross_val_score(
        model, X, y, cv=5, scoring="roc_auc"
    ).mean()

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"CV ROC-AUC: {cv_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    if auc > best_auc:
        best_auc = auc
        best_model = model

# -----------------------------
# Save model & feature schema
# -----------------------------
os.makedirs("../models", exist_ok=True)

dump(best_model, "../models/titanic_model.joblib")
dump(feature_columns, "../models/feature_columns.joblib")

print(f"\n✅ Best model saved with ROC-AUC: {best_auc:.4f}")
