"""Shared settings for the project.

By default the project uses SQLite (zero setup).
To use PostgreSQL instead, set the DATABASE_URL environment variable, e.g.
    postgresql+psycopg2://user:password@localhost:5432/churn_db
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "churn_data.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
IMPORTANCE_PATH = BASE_DIR / "model" / "feature_importance.csv"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'churn.db'}")

TARGET = "Churn"          # column with Yes/No values
ID_COLUMN = "customerID"  # identifier column (not used for training)
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.35
