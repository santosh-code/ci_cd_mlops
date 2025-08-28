from fastapi.testclient import TestClient
from src.inference import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Iris Classifier API is running!"}

def test_prediction():
    features = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)