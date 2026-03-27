import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

from preprocess import load_and_preprocess

ROLL_NO = "2022BCS0208"
NAME = "Sanjana"

df = load_and_preprocess("data/dataset.csv")

# Feature selection (change for runs)
FEATURES = ['no_of_dependents','education','self_employed','income_annum',
            'loan_amount','loan_term','cibil_score',
            'residential_assets_value','commercial_assets_value',
            'luxury_assets_value','bank_asset_value']

X = df[FEATURES]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment(f"{ROLL_NO}_experiment")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Save model
    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    # Save metrics JSON (MANDATORY)
    with open("metrics.json", "w") as f:
        import json
        json.dump({
            "accuracy": acc,
            "f1_score": f1,
            "Name": NAME,
            "Roll No": ROLL_NO
        }, f)
