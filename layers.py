import numpy as np


class Layer:
    """
    Base class for layers
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        Forward pass
        """
        return input

    def backward(self, input, grad_output):
        """
        Backward pass
        """
        units = input.shape[1]
        return grad_output
