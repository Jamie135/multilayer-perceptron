import numpy as np


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
    Calculate the binary cross entropy loss of the model.
    - output: predicted results of the model in 2D array of shape (batch, classes) 
    - y: expected results in 1D array of shape (batch,) 
    """
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)

    # transform y with one-hot matrix to have the same shape as output
    y_one_hot = np.zeros_like(output)
    y_one_hot[np.arange(len(output)), y] = 1

    loss = - (y_one_hot * np.log(output) + (1 - y_one_hot) * np.log(1 - output))
    return np.mean(loss, axis=-1)


def compute_gradients(output, y):
    """
    Calculate the gradients of the loss from on the output of the model
    
    - output: predicted results of the model in 2D array of shape (batch, classes)
    - y: expected results in 1D array of shape (batch,) 
    """

    # create an array of zeros with the same shape as the output
    one_hot = np.zeros_like(output)

    # set the value of the correct answers to 1 in the one_hot array
    one_hot[np.arange(len(output)), y] = 1

    softmax = np.exp(output) / np.exp(output).sum(axis=-1, keepdims=True)
    return (- one_hot + softmax) / output.shape[0]


def scores(layers, X, phase):
    """
    Performs a forward propagation through the layers
    to compute the raw output scores for each input (X) sample
    by finding the index of the maximum score for each sample
    """

    output = forward_propagation(layers, X)[-1]
    if phase == 'train':
        return output.argmax(axis=-1)
    elif phase == 'predict':
        return output
    else:
        raise ValueError(f"Invalid phase: {phase}")



