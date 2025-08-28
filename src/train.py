import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess

def train_model():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2f}")

    # save model to root/models/
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model()
