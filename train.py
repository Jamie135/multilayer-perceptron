import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from layers import *
from propagation import *


def parse():
    """
    Parse arguments
    """

    def valid_layers(value):
        ivalue = int(value)
        if ivalue < 2 or ivalue > 50:
            raise argparse.ArgumentTypeError(f"{value} is an invalid value")
        return ivalue
    
    def valid_epochs(value):
        ivalue = int(value)
        if ivalue < 1 or ivalue > 10000:
            raise argparse.ArgumentTypeError(f"{value} is an invalid value")
        return ivalue
    
    def valid_learning(value):
        fvalue = float(value)
        if fvalue < 1 and fvalue > 0:
            return fvalue
        else:
            raise argparse.ArgumentTypeError(f"{value} is an invalid value")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", type=valid_layers, help="Number of hidden layers")
    parser.add_argument("-e", "--epochs", type=valid_epochs, default=1000, help="Number of hidden layers")
    parser.add_argument("-lr", "--learning", type=valid_learning, default=0.001, help="Number of hidden layers")
    parser.add_argument("-a", "--hidden", type=str, default="relu", help="Activation function in the hidden layers")
    parser.add_argument("-es", "--earlystop", type=bool, default=False, help="Early stopping")
    args = parser.parse_args()
    return args


def preprocess(data_train, data_test):
    """
    Normalize data to prevent overflows
    Label encode data to transform string into 0 or 1
    """

    # Normalization
    X_train = data_train.iloc[:, 2:]
    X_test = data_test.iloc[:, 2:]

    scaler = StandardScaler()

    X_train_normalized = scaler.fit_transform(X_train.values)
    X_test_normalized = scaler.fit_transform(X_test.values)

    # Label encoding
    y_train = data_train.iloc[:, 1].values
    y_test = data_test.iloc[:, 1].values

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    return X_train_normalized, X_test_normalized, y_train, y_test


def create_layers(X, hidden, learning_rate, layers=None):
    """
    Create the layers of the model
    """

    network = []
    activations = {
        "relu": ReLU,
        "leakyrelu": LeakyReLU,
        "sigmoid": Sigmoid
    }

    if hidden not in activations:
        raise ValueError(f"Activation function not supported: {hidden}")

    network.append(Dense(X.shape[1], 64, learning_rate))
    network.append(activations[hidden](learning_rate))

    if layers:
        layers_number = layers
    if hidden == "relu" and layers == None:
        layers_number = 9
    elif (hidden == "leakyrelu" or hidden == "sigmoid") and layers == None:
        layers_number = 2

    for _ in range(layers_number - 1):
        network.append(Dense(64, 64, learning_rate))
        network.append(activations[hidden](learning_rate))

    network.append(Dense(64, 2, learning_rate))
    network.append(Softmax())

    return network


def minibatches(X, y, batchsize=16):
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


def save_model(layers, hidden, learning_rate):
    """
    Save the model to a JSON file
    """

    model = {}
    model['layers'] = []
    for l in layers:
        if isinstance(l, Dense):
            layer = {
                'units': l.input_unit,
                'activation': hidden,
                'weights': l.weights.tolist(),
                'biases': l.biases.tolist()
            }
            model['layers'].append(layer)
    model['learning_rate'] = learning_rate
    with open('model.json', 'w') as f:
        json.dump(model, f)


def train(args: str = None):
    """
    Train the model
    """

    if not (os.path.isfile("data/data_training.csv") or os.path.isfile("data/data_test.csv")):
        raise FileNotFoundError(f"Dataset files not found.")

    data_train = pd.read_csv('data/data_training.csv', header=None)
    data_test = pd.read_csv('data/data_test.csv', header=None)

    X_train, X_test, y_train, y_test = preprocess(data_train, data_test)

    # print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    layers = create_layers(X_train, args.hidden, args.learning, args.layers)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []

    epochs = args.epochs

    # Initialize live plot
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[2].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy')
    ax[2].set_ylabel('F1-Score')

    best_val_loss = float('inf')
    patience_counter = 0
    for i in range(epochs):
        for X_batch, y_batch in minibatches(X_train, y_train):
            propagation(layers, X_batch, y_batch)
        # propagation(layers, X_train, y_train)

        train_loss.append(propagation(layers, X_train, y_train))
        test_loss.append(propagation(layers, X_test, y_test))
        train_acc.append(np.mean(scores(layers, X_train, phase='train') == y_train))
        test_acc.append(np.mean(scores(layers, X_test, phase='train') == y_test))

        y_train_pred = scores(layers, X_train, phase='train')
        y_test_pred = scores(layers, X_test, phase='train')
        train_f1.append(f1_score(y_train, y_train_pred, average='weighted'))
        test_f1.append(f1_score(y_test, y_test_pred, average='weighted'))

        print(f"Epoch {i + 1}/{epochs}")
        print("Train loss:", train_loss[-1])
        print("Validation loss:", test_loss[-1])
        print("Train accuracy:", train_acc[-1])
        print("Validation accuracy:", test_acc[-1])
        print("\n")

        if args.earlystop:
            if test_loss[-1] < best_val_loss:
                best_val_loss = test_loss[-1]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 250:
                print("Early stopping triggered")
                break

        # Update live plot
        if i % 10 == 0:
            ax[0].plot(train_loss, 'b-', label='train loss' if i == 0 else "")
            ax[0].plot(test_loss, 'r-', label='test loss' if i == 0 else "")
            ax[1].plot(train_acc, 'b-', label='train accuracy' if i == 0 else "")
            ax[1].plot(test_acc, 'r-', label='test accuracy' if i == 0 else "")
            ax[2].plot(range(len(train_f1)), train_f1, label='Train F1-Score')
            ax[2].plot(range(len(test_f1)), test_f1, label='Test F1-Score')
            if i == 0:
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    try:
        os.remove("model.json")
    except OSError:
        pass
    save_model(layers, args.hidden, args.learning)


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
