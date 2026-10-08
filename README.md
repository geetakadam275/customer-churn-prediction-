# Customer Churn Prediction & Business Analytics System

Predicts which customers are likely to leave, stores the results in a database,
and shows everything in an easy dashboard.

**Team:** Gita Parmeshwar Kadam (23022521242057), Asmita Balaji Raut (23022521242059),
Keshav Gajendra Rajurkar (23022521242058)

## How it works

```
data/churn_data.csv  ->  train_model.py  ->  database (customers table)  ->  app.py (dashboard)
                              |
                       model/churn_model.joblib
```

| File | What it does |
|------|--------------|
| `generate_data.py` | Creates a sample customer dataset (skip if you have a real one) |
| `train_model.py` | Cleans data, trains Logistic Regression, Decision Tree and Random Forest, picks the best, saves predictions to the database |
| `app.py` | Streamlit dashboard: KPIs, churn by segment, model scores, high-risk list, single-customer prediction |
| `config.py` | File paths, database URL, risk thresholds |

## Run it (4 commands)

```bash
pip install -r requirements.txt
python generate_data.py      # step 1: sample data (or put your own CSV in data/churn_data.csv)
python train_model.py        # step 2: train models + fill database
streamlit run app.py         # step 3: open the dashboard in your browser
```

## Using a real dataset

Download the "Telco Customer Churn" CSV from Kaggle, save it as `data/churn_data.csv`
and skip `generate_data.py`. It needs a `Churn` column (Yes/No) and a `customerID` column;
all other columns are picked up automatically.

## Using PostgreSQL (optional)

By default a local SQLite file (`churn.db`) is used, so nothing needs installing.
To use PostgreSQL, install `psycopg2-binary`, create an empty database, then set:

```bash
# Windows (PowerShell)
$env:DATABASE_URL="postgresql+psycopg2://postgres:yourpassword@localhost:5432/churn_db"
# Linux / Mac
export DATABASE_URL="postgresql+psycopg2://postgres:yourpassword@localhost:5432/churn_db"
```

Then run `train_model.py` and `app.py` as usual.

## Project steps (for your report)

1. **Data collection** - customer tenure, charges, contract, payment method, services, support calls
2. **Preprocessing** - remove duplicates, fix missing values, encode categories, scale numbers
3. **EDA** - churn rate by contract, payment method, tenure group (dashboard Overview tab)
4. **Modelling** - Logistic Regression, Decision Tree, Random Forest; compared using accuracy, precision, recall, F1, ROC-AUC
5. **Storage** - customer data and churn probability saved in the database
6. **Dashboard** - business metrics, high-risk customers and live prediction
