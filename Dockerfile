FROM python:3.10-slim

WORKDIR /app

# Copy only requirements for inference
COPY requirements.txt .

# Install only what is needed for inference
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy scikit-learn

# Copy inference code and model
COPY src/inference.py ./src/inference.py
COPY models/ ./models

EXPOSE 5000

CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "5000"]
