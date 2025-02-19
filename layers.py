import numpy as np


class Layer:
    """
    Base class for layers that contains basic methods
    """

    def __init__(self):
        """
        Base constructor
        """
        pass


    def forward(self, input):
        """
        Base forward propagation that returns the exact input
        """
        return input


    def backward(self, input, grad_output):
        """
        Base backward propagation that transform the gradient of the loss into the shape of the input
        """
        units = input.shape[1]
        # np.eye createxs an identity matrix of shape (units, units)
        id_matrix = np.eye(units)
        return np.dot(grad_output, id_matrix)


class Dense(Layer):
    """
    Hidden layer that applies affine transformation
    """

    def __init__(self, input, output, learning_rate=0.01, initialization="he"):
        """
        Constructor of the Dense class that initializes the weights, biases and learning rate
        """
        if initialization == "he":
            limit = np.sqrt(6 / input)
        elif initialization == "lecun":
            limit = np.sqrt(3 / input)
        else:
            raise ValueError("Invalid initialization,use 'he' or 'lecun'")

        self.weights = np.random.uniform(-limit, limit, size=(input, output))
        self.biases = np.zeros(output)
        self.learning_rate = learning_rate
    

    def forward(self, input):
        """
        Perform an affine transformation
        """
        return np.dot(input, self.weights) + self.biases
    

    def backward(self, input, grad_output):
        """
        Compute gradients and update weights and biases then returns the gradient of the loss on the input
        """
        # calculate the gradients
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # update the weights and biases
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input


class Sigmoid(Layer):
    """
    Hidden layer that applies the sigmoid activation function
    """

    def __init__(self):
        """
        Constructor of the Sigmoid class
        """
        pass


    def forward(self, input):
        """
        Apply the sigmoid function
        """
        return 1 / (1 + np.exp(-input))
    

    def backward(self, input, grad_output):
        """
        Compute the gradient of the loss on the input
        """
        A = 1 / (1 + np.exp(-input))
        return grad_output * A * (1 - A)


class ReLU(Layer):
    """
    Hidden layer that applies the ReLU activation function (Rectified Linear Unit)
    This activation function is used to replace all the negative values in the input by 0
    """

    def __init__(self):
        """
        Constructor of the ReLU class
        """
        pass


    def forward(self, input):
        """
        Apply the ReLU function
        """
        return np.maximum(0, input)
    

    def backward(self, input, grad_output):
        """
        Compute the gradient of the loss on the input
        """
        return grad_output * (input > 0)
