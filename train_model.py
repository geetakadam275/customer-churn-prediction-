"""Step 2: clean the data, train 3 models, save the best one and store results in the database."""
import json

import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (DATA_PATH, DATABASE_URL, HIGH_RISK_THRESHOLD, ID_COLUMN, IMPORTANCE_PATH,
                    MEDIUM_RISK_THRESHOLD, METRICS_PATH, MODEL_PATH, TARGET)


def load_and_clean(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates()                               # remove duplicate records
    print(f"Removed {before - len(df)} duplicate rows")
    if "TotalCharges" in df.columns:                        # blanks in this column are common
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df.reset_index(drop=True)


def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def risk_level(p):
    if p >= HIGH_RISK_THRESHOLD:
        return "High"
    if p >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def main():
    if not DATA_PATH.exists():
        import generate_data
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted[name] = pipe
        print(f"{name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in results[name].items()))

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Save model + metrics
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"best_model": best_name, "results": results}, indent=2))

    # Feature importance (which factors drive churn)
    names = best.named_steps["prep"].get_feature_names_out()
    model = best.named_steps["model"]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
    imp = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "importance": values})
    imp.sort_values("importance", ascending=False).to_csv(IMPORTANCE_PATH, index=False)

    # Score every customer and store everything in the database
    out = df.copy()
    out["churn_probability"] = best.predict_proba(X)[:, 1].round(4)
    out["risk_level"] = out["churn_probability"].apply(risk_level)
    engine = create_engine(DATABASE_URL)
    out.to_sql("customers", engine, if_exists="replace", index=False)
    print(f"Saved {len(out)} scored customers to the database table 'customers'")


if __name__ == "__main__":
    main()
