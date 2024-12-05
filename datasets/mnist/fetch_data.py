from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def fetch_data():
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy().astype(int)
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]

    return X_train, X_test, y_train, y_test
