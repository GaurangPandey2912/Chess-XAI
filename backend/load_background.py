import pandas as pd
import numpy as np

def load_background(n=100):
    df = pd.read_csv("data/positions.csv")
    X = df.iloc[:, :-1].values.astype(np.float32)
    idx = np.random.choice(len(X), n, replace=False)
    return X[idx]
