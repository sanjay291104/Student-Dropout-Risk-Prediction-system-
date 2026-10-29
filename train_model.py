"""
train_model.py
Student Dropout Risk Prediction - Model Training Script
----------------------------------------------------------
Trains EXISTING SYSTEM models : Logistic Regression, Decision Tree, KNN,
                                 Naive Bayes, SVM, ANN (MLPClassifier)
Trains PROPOSED SYSTEM models : Decision Tree, Random Forest, XGBoost

Saves:
  models/best_model.pkl        -> best performing proposed model
  models/scaler.pkl            -> StandardScaler fitted on numeric columns
  models/encoders.pkl          -> dict of LabelEncoders for categorical columns
  models/feature_columns.pkl   -> ordered list of feature columns model expects
  models/metrics.json          -> accuracy/precision/recall/f1 for every model
  models/feature_importance.json -> feature importance of the best model
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "student_dropout_dataset.csv"

CATEGORICAL_COLS = ["Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
                     "Semester", "Department", "Parental_Education"]
NUMERIC_COLS = ["Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
                 "Assignment_Delay_Days", "Travel_Time_Minutes", "Stress_Index",
                 "GPA", "Semester_GPA", "CGPA"]
TARGET_COL = "Dropout"
DROP_COLS = ["Student_ID"]


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Impute numeric missing values with column median
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical missing values with mode
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categoricals(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    print(f"[{name:22s}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm,
    }


def main():
    print("Loading and cleaning dataset...")
    df = load_and_clean(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df[TARGET_COL].value_counts())

    df_enc, encoders = encode_categoricals(df, fit=True)

    feature_columns = CATEGORICAL_COLS + NUMERIC_COLS
    X = df_enc[feature_columns]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metrics = {"existing_system": {}, "proposed_system": {}}

    # ---------------- EXISTING SYSTEM MODELS ----------------
    print("\n===== EXISTING SYSTEM MODELS =====")

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    metrics["existing_system"]["Logistic Regression"] = evaluate("Logistic Regression", lr, X_test_scaled, y_test)

    dt_existing = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    dt_existing.fit(X_train_scaled, y_train)
    metrics["existing_system"]["Decision Tree"] = evaluate("Decision Tree (existing)", dt_existing, X_test_scaled, y_test)

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train_scaled, y_train)
    metrics["existing_system"]["KNN"] = evaluate("KNN", knn, X_test_scaled, y_test)

    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    metrics["existing_system"]["Naive Bayes"] = evaluate("Naive Bayes", nb, X_test_scaled, y_test)

    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    metrics["existing_system"]["SVM"] = evaluate("SVM", svm, X_test_scaled, y_test)

    ann = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=RANDOM_STATE)
    ann.fit(X_train_scaled, y_train)
    metrics["existing_system"]["ANN"] = evaluate("ANN (MLP)", ann, X_test_scaled, y_test)

    # ---------------- PROPOSED SYSTEM MODELS ----------------
    print("\n===== PROPOSED SYSTEM MODELS =====")

    dt_proposed = DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=RANDOM_STATE)
    dt_proposed.fit(X_train_scaled, y_train)
    metrics["proposed_system"]["Decision Tree"] = evaluate("Decision Tree (proposed)", dt_proposed, X_test_scaled, y_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    metrics["proposed_system"]["Random Forest"] = evaluate("Random Forest", rf, X_test_scaled, y_test)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train)
    metrics["proposed_system"]["XGBoost"] = evaluate("XGBoost", xgb, X_test_scaled, y_test)

    # ---------------- SELECT BEST PROPOSED MODEL ----------------
    proposed_models = {"Decision Tree": dt_proposed, "Random Forest": rf, "XGBoost": xgb}
    best_name = max(metrics["proposed_system"], key=lambda k: metrics["proposed_system"][k]["f1_score"])
    best_model = proposed_models[best_name]
    print(f"\nBest proposed model selected: {best_name}")

    metrics["best_model"] = best_name

    # Feature importance (if available)
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_columns, best_model.feature_importances_.tolist()))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    else:
        importances = {}

    # ---------------- SAVE ARTIFACTS ----------------
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("models/feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    print("\nAll artifacts saved to /models")


if __name__ == "__main__":
    main()
