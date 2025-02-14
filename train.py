import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from split import split


def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--options")
    args = parser.parse_args()
    return args


def format(data_train, data_test):
    # Normalization
    X_train = data_train.iloc[:, 2:].values
    X_test = data_test.iloc[:, 2:].values

    scaler = MinMaxScaler()

    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Label encoding
    y_train = data_train.iloc[:, 1].values
    y_test = data_test.iloc[:, 1].values

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = label_encoder.transform(y_test)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # Transpose
    X_train_normalized = X_train_normalized.T
    X_test_normalized = X_test_normalized.T
    y_train = y_train.T
    y_test = y_test.T

    return X_train_normalized, X_test_normalized, y_train, y_test


def train(args: str = None):
    if not (os.path.isfile("data_train.csv") or os.path.isfile("data_test.csv")):
        raise FileNotFoundError(f"Dataset files not found.")

    data_train = pd.read_csv('data_train.csv', header=None)
    data_test = pd.read_csv('data_test.csv', header=None)

    X_train, X_test, y_train, y_test = format(data_train, data_test)

    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")









def main():
    try:
        args = parse()
        train(args)
    except FileNotFoundError as f:
        print(f)
        sys.exit(1)
    except ValueError as v:
        print(v)
        sys.exit(1)

if __name__ == "__main__":
    main()
