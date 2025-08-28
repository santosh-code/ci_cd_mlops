import pandas as pd
from sklearn.datasets import load_iris

def load_and_preprocess():
    # Load iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(X.head())
    print(y.head())
