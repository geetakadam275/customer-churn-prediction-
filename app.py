"""Step 3: the business dashboard.  Run with:  streamlit run app.py"""
import json

import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from config import (DATABASE_URL, HIGH_RISK_THRESHOLD, ID_COLUMN, IMPORTANCE_PATH,
                    MEDIUM_RISK_THRESHOLD, METRICS_PATH, MODEL_PATH, TARGET)

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction & Business Analytics")


@st.cache_data
def load_customers():
    return pd.read_sql("SELECT * FROM customers", create_engine(DATABASE_URL))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_customers()
except Exception:
    st.error("No data found. Run `python generate_data.py` and `python train_model.py` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Model performance", "High-risk customers", "Predict a customer"])

# ---------------------------------------------------------------- Overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df[TARGET].mean():.1%}")
    if "MonthlyCharges" in df:
        c3.metric("Avg monthly charges", f"{df['MonthlyCharges'].mean():.2f}")
    c4.metric("High-risk customers", f"{(df['risk_level'] == 'High').sum():,}")

    st.subheader("Churn rate by customer segment")
    cats = [c for c in df.select_dtypes(exclude="number").columns
            if c not in (ID_COLUMN, "risk_level") and df[c].nunique() <= 10]
    if "tenure" in df:
        df["tenure_group"] = pd.cut(df["tenure"], [0, 12, 24, 48, 100],
                                    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"])
        cats.append("tenure_group")
    choice = st.selectbox("Group customers by", cats)
    rate = df.groupby(choice, observed=True)[TARGET].mean().mul(100).round(1)
    st.bar_chart(rate)
    st.caption("Churn rate (%) for each group")

    st.subheader("Risk levels")
    st.bar_chart(df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]))

# ---------------------------------------------------------- Model performance
with tab2:
    m = json.loads(METRICS_PATH.read_text())
    st.write(f"**Best model:** {m['best_model']}  (chosen by ROC-AUC)")
    st.dataframe(pd.DataFrame(m["results"]).T.round(3))
    st.subheader("What drives churn? (top features)")
    imp = pd.read_csv(IMPORTANCE_PATH).head(10).set_index("feature")
    st.bar_chart(imp)

# ------------------------------------------------------- High-risk customers
with tab3:
    st.write("Customers most likely to leave, so the retention team can contact them first.")
    level = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    risky = (df[df["risk_level"].isin(level)]
             .sort_values("churn_probability", ascending=False))
    st.write(f"{len(risky):,} customers")
    st.dataframe(risky, width="stretch")
    st.download_button("Download as CSV", risky.to_csv(index=False), "high_risk_customers.csv")

# --------------------------------------------------------- Predict a customer
with tab4:
    st.write("Enter customer details to get a churn prediction.")
    model = load_model()
    features = df.drop(columns=[TARGET, ID_COLUMN, "churn_probability", "risk_level", "tenure_group"],
                       errors="ignore")
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(features.columns):
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(features[col]):
                values[col] = st.number_input(col, float(features[col].min()),
                                              float(features[col].max()),
                                              float(features[col].median()))
            else:
                values[col] = st.selectbox(col, sorted(features[col].dropna().unique()))
    if st.button("Predict"):
        p = model.predict_proba(pd.DataFrame([values]))[0, 1]
        level = ("High" if p >= HIGH_RISK_THRESHOLD else
                 "Medium" if p >= MEDIUM_RISK_THRESHOLD else "Low")
        st.metric("Churn probability", f"{p:.1%}")
        st.write(f"**Risk level: {level}**")
