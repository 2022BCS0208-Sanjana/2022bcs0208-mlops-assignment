import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

ROLL_NO = "2022BCS0208"
NAME = "Sanjana"

DATA_PATH = "data/dataset_v1.csv"   # change per run
USE_REDUCED_FEATURES = False
N_ESTIMATORS = 100

df = pd.read_csv(DATA_PATH)

if USE_REDUCED_FEATURES:
    FEATURES = ['income_annum', 'loan_amount', 'cibil_score']
else:
    FEATURES = ['no_of_dependents','education','self_employed','income_annum',
                'loan_amount','loan_term','cibil_score',
                'residential_assets_value','commercial_assets_value',
                'luxury_assets_value','bank_asset_value']

X = df[FEATURES]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment(f"{ROLL_NO}_experiment")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("dataset", DATA_PATH)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("features_used", len(FEATURES))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "f1_score": f1,
            "Name": NAME,
            "Roll No": ROLL_NO
        }, f)
