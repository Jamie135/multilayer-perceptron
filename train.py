import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from layers import *


def parse():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser()
    # parser.add_argument("--options")
    args = parser.parse_args()
    return args


def preprocess(data_train, data_test):
    """
    Normalize data to prevent overflows
    Label encode data to transform string into 0 or 1
    """

    # Normalization
    X_train = data_train.drop(data_train.columns[1], axis=1)
    X_test = data_test.drop(data_test.columns[1], axis=1)

    scaler = MinMaxScaler()

    X_train_normalized = scaler.fit_transform(X_train.values)
    X_test_normalized = scaler.fit_transform(X_test.values)

    # Label encoding
    y_train = data_train.iloc[:, 1].values
    y_test = data_test.iloc[:, 1].values

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    return X_train_normalized, X_test_normalized, y_train, y_test


def propagation(layers, X, y):
    """
    Use forward and backward propagation to train the model
    """

    activations = forward_propagation(layers, X)
    loss = backward_propagation(layers, activations, X, y)
    return loss


def forward_propagation(layers, X):
    """
    Apply activation functions to the hidden layers
    """

    activations = []
    input = X
    for l in layers:
        activations.append(l.forward(input))
        input = activations[-1]
    return activations


def backward_propagation(layers, activations, X, y):
    """
    Update the weights of the model by propagating the gradients in backward
    """

    input = [X] + activations
    output = activations[-1]

    loss = loss(output, y)
    gradients = gradients(output, y)

    for l in range(len(layers))[::-1]:
        layer = layers[l]
        gradients = layer.backward(input[l], gradients)
    
    return np.mean(loss)


def loss(output, y):
    """
    Calculate the loss of the model using softmax and logs
    The purpose of log is to prevent overflows in exponential

    - current_output: output of the model in 2D array of shape (batch, classes)
    - y: expected results in 1D array of shape (batch,) 
    """

    # get the scores corresponding to the correct answers
    target_scores = output[np.arange(len(output)), y]

    cross_entropy = - target_scores + np.log(np.sum(np.exp(output), axis=-1))
    return cross_entropy


def gradients(output, y):
    """
    Calculate the gradients of the loss from on the output of the model
    
    - current_output: output of the model in 2D array of shape (batch, classes)
    - y: expected results in 1D array of shape (batch,) 
    """

    # create an array of zeros with the same shape as the output
    one_hot = np.zeros_like(output)

    # set the value of the correct answers to 1 in the one_hot array
    one_hot[np.arange(len(output)), y] = 1

    softmax = np.exp(output) / np.exp(output).sum(axis=-1, keepdims=True)
    return (- one_hot + softmax) / output.shape[0]


def train(args: str = None):
    if not (os.path.isfile("data_train.csv") or os.path.isfile("data_test.csv")):
        raise FileNotFoundError(f"Dataset files not found.")

    data_train = pd.read_csv('data_train.csv', header=None)
    data_test = pd.read_csv('data_test.csv', header=None)

    X_train, X_test, y_train, y_test = preprocess(data_train, data_test)

    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    layers = []


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
