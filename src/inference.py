import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Correct model path (project root/models/model.pkl)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

app = FastAPI(title="Iris Classifier API", version="1.0")

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
