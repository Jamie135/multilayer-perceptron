import os
import sys
import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
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

    loss = compute_loss(output, y)
    gradients = compute_gradients(output, y)

    for l in range(len(layers))[::-1]:
        layer = layers[l]
        gradients = layer.backward(input[l], gradients)
    
    return np.mean(loss)


def compute_loss(output, y):
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


def compute_gradients(output, y):
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


def minibatches(X, y, batchsize=32):
    """
    Create minibatches of the dataset
    """

    assert len(X) == len(y)

    indices = np.random.permutation(len(X))
    for i in range(0, len(X) - batchsize + 1, batchsize):
        suffled_indices = indices[i:i + batchsize]
        # yield returns a tuple containing the inputs and targets for the current minibatch
        # this allows the function to be used as a generator, producing one minibatch at a time
        yield X[suffled_indices], y[suffled_indices]


def predict(layers, X):
    """
    Performs a forward propagation through the layers
    to compute the raw output scores for each input (X) sample
    by finding the index of the maximum score for each sample
    """

    output = forward_propagation(layers, X)[-1]
    return output.argmax(axis=-1)


def train(args: str = None):
    """
    Train the model
    """

    if not (os.path.isfile("data_train.csv") or os.path.isfile("data_test.csv")):
        raise FileNotFoundError(f"Dataset files not found.")

    data_train = pd.read_csv('data_train.csv', header=None)
    data_test = pd.read_csv('data_test.csv', header=None)

    X_train, X_test, y_train, y_test = preprocess(data_train, data_test)

    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    layers = []
    layers.append(Dense(X_train.shape[1], 50))
    layers.append(ReLU())
    layers.append(Dense(50, 100))
    layers.append(ReLU())
    layers.append(Dense(100, 2))
    layers.append(Sigmoid())

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for i in range(2500):
        for X_batch, y_batch in minibatches(X_train, y_train, batchsize=32):
            propagation(layers, X_batch, y_batch)
    
        train_loss.append(propagation(layers, X_train, y_train))
        test_loss.append(propagation(layers, X_test, y_test))
        train_acc.append(np.mean(predict(layers, X_train) == y_train))
        test_acc.append(np.mean(predict(layers, X_test) == y_test))

        print("Epoch", i + 1)
        print("Train loss:", train_loss[-1])
        print("Validation loss:", test_loss[-1])
        print("Train accuracy:", train_acc[-1])
        print("Validation accuracy:", test_acc[-1])
        print("\n")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.legend()
    plt.show()


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
