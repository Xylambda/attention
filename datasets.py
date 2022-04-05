import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchfitter.utils.preprocessing import tabular_to_sliding_dataset

np.random.seed(0)


def generate_venezia_high_waters():
    data = pd.read_pickle("data/venezia_high_waters.pkl")
    data = data.groupby(data.index.date).mean()

    val_idx = int(len(data) * 0.7)
    tst_idx = int(len(data) * 0.8)

    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        data.values,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=20,
        n_future=1,
        scaler=StandardScaler()
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_sine_waves():
    _range = np.arange(-20, 20, 0.01)
    _iter = [_range / (np.pi / x) for x in range(1, 7)]
    _range = np.stack(_iter, axis=1)
    sine = np.sin(_range)

    val_idx = int(len(sine) * 0.7)
    tst_idx = int(len(sine) * 0.8)

    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        sine,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=20,
        n_future=1
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_white_noise():
    random_walk = np.random.normal(0, 0.01, (4_000, 4))

    val_idx = int(len(random_walk) * 0.7)
    tst_idx = int(len(random_walk) * 0.8)

    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        random_walk,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=20,
        n_future=1
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_stock_returns():
    data = pd.read_pickle("data/XAUUSD.pkl")
    data = data.pct_change().dropna()

    val_idx = int(len(data) * 0.7)
    tst_idx = int(len(data) * 0.8)

    X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
        data,
        validation_idx=val_idx,
        test_idx=tst_idx,
        n_past=20,
        n_future=1
    )

    return X_train, y_train, X_val, y_val, X_test, y_test