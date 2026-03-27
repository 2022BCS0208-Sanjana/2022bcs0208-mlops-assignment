from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

ROLL_NO = "2022BCS0208"
NAME = "Sanjana"

@app.get("/")
def health():
    return {
        "Name": NAME,
        "Roll No": ROLL_NO
    }

@app.post("/predict")
def predict(data: dict):
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(values)[0]

    return {
        "prediction": int(prediction),
        "Name": NAME,
        "Roll No": ROLL_NO
    }
