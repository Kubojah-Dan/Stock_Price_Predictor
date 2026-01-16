import numpy as np
from sklearn.model_selection import train_test_split

def time_split(X, y, test_size):
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]

