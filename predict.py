import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from layers import *
from propagation import *


def parse():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def preprocess(data_test):
    """
    Normalize data to prevent overflows
    Label encode data to transform string into 0 or 1
    """

    data_test = data_test.replace('B', '0', regex=True)
    data_test = data_test.replace('M', '1', regex=True)

    scaler = StandardScaler()
    X_test = data_test.iloc[:, 2:]
    X_test = scaler.fit_transform(X_test)

    y_test = data_test.iloc[:, 1]
    y_test = y_test.to_numpy().astype(int)

    return X_test, y_test


def load_model(filepath='model.json'):
    """
    Load the model from a JSON file
    """

    with open(filepath, 'r') as f:
        model = json.load(f)
    return model


def create_layers(model):
    """
    Create the layers of the model
    """

    layers = []
    activations = {
        "leakyrelu": LeakyReLU,
        "relu": ReLU,
        "softmax": Softmax,
        "sigmoid": Sigmoid
    }
    learning_rate = model['learning_rate']

    for layer in model['layers']:
        layers.append(Dense(layer['units'], len(layer['weights'][0]), learning_rate))
        layers[-1].weights = np.array(layer['weights'])
        layers[-1].biases = np.array(layer['biases'])
        layers.append(activations[layer['activation']](learning_rate))
    
    return layers


def softmax(logits):
    """
    Apply softmax to logits to get probabilities
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(probabilities, y_test):
    """
    Compute the cross entropy loss
    """

    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1. - epsilon)
    log_likelihood = -np.log(probabilities[range(y_test.shape[0]), y_test])
    loss = np.sum(log_likelihood) / y_test.shape[0]
    return loss


def predict(args: str = None):
    """
    Predict the output of the model
    """

    data_test = pd.read_csv('data/data_test.csv', header=None)

    X_test, y_test = preprocess(data_test)

    model = load_model()
    layers = create_layers(model)

    logits = scores(layers, X_test, phase='predict')
    probabilities = softmax(logits)

    y_pred = np.argmax(probabilities, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('Accuracy: {0:.4f}%'.format(((tn + tp) / y_test.shape[0])*100))
    print('Binary Cross-Entropy Loss: {0:.4f}\n'.format(cross_entropy_loss(probabilities, y_test)))


def main():
    try:
        args = parse()
        predict(args)
    except FileNotFoundError as f:
        print(f)
        sys.exit(1)
    except ValueError as v:
        print(v)
        sys.exit(1)


if __name__ == '__main__':
    main()