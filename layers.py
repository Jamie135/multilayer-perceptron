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

    def __init__(self, input_unit, output_unit, learning_rate, initialization="he"):
        """
        Constructor of the Dense class that initializes the weights, biases and learning rate
        """
        if initialization == "he":
            limit = np.sqrt(6 / input_unit)
        elif initialization == "lecun":
            limit = np.sqrt(3 / output_unit)
        else:
            raise ValueError("Invalid initialization,use 'he' or 'lecun'")

        np.random.seed(2)
        self.input_unit = input_unit
        self.weights = np.random.uniform(-limit, limit, size=(input_unit, output_unit))
        self.biases = np.zeros(output_unit)
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

    def __init__(self, learning_rate):
        """
        Constructor of the Sigmoid class
        """
        self.learning_rate = learning_rate
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

    def __init__(self, learning_rate):
        """
        Constructor of the ReLU class
        """
        self.learning_rate = learning_rate
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


class LeakyReLU(Layer):
    """
    Hidden layer that applies the Leaky ReLU activation function
    This activation function allows a small gradient when the input is negative
    """

    def __init__(self, learning_rate):
        """
        Constructor of the LeakyReLU class
        :param learning_rate: Slope of the function when input is negative
        """
        self.learning_rate = learning_rate


    def forward(self, input):
        """
        Apply the Leaky ReLU function
        :param input: Input data
        :return: Transformed data
        """
        return np.where(input > 0, input, self.learning_rate * input)


    def backward(self, input, grad_output):
        """
        Compute the gradient of the loss on the input
        :param input: Input data
        :param grad_output: Gradient of the loss with respect to the output
        :return: Gradient of the loss with respect to the input
        """
        grad_input = np.ones_like(input)
        grad_input[input <= 0] = self.learning_rate
        return grad_output * grad_input


class Softmax(Layer):
    """
    Output layer that applies the softmax activation function
    """

    def __init__(self):
        """
        Constructor of the Softmax class
        """
        pass


    def forward(self, input):
        """
        Apply the softmax function
        """
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


    def backward(self, input, grad_output):
        """
        Compute the gradient of the loss on the input
        """
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        n = input.shape[0]
        dinput = np.empty_like(input)
        for i in range(n):
            p = probabilities[i]
            jacobian_matrix = np.diag(p) - np.outer(p, p)
            dinput[i] = np.dot(jacobian_matrix, grad_output[i])
        return dinput
