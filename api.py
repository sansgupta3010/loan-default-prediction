from fastapi import FastAPI
import joblib

app = FastAPI(
    title="Loan Default Prediction API"
)

model = joblib.load("loan_default_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Loan Default Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    prediction = model.predict([data])[0]

    return {
        "prediction": int(prediction),
        "default": "Yes" if prediction == 1 else "No"
    }