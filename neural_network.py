import numpy as np
import math

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, numInputs, numHidden, numOutputs):
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs

        self.weights_ih = np.random.uniform(low = -1.0, high = 1.0, size=(self.numHidden, self.numInputs))
        self.weights_ho = np.random.uniform(low = -1.0, high = 1.0, size=(self.numOutputs, self.numHidden))

        self.biases_h = np.random.uniform(low = -1.0, high = 1.0, size=(self.numHidden, 1))
        self.biases_o = np.random.uniform(low = -1.0, high = 1.0, size=(self.numOutputs, 1))

        self.learning_rate = 0.2

    def feed_forward(self, inputs):
        sigmoid_v = np.vectorize(sigmoid)
        inputs = np.array(inputs)
        inputs = np.reshape(inputs, (self.numInputs, 1))

        hidden = np.matmul(self.weights_ih, inputs) + self.biases_h
        hidden = sigmoid_v(hidden)

        output = np.matmul(self.weights_ho, hidden) + self.biases_o
        output = sigmoid_v(output)

        return output.tolist()

    def train(self, inputs, targets):
        sigmoid_v = np.vectorize(sigmoid)
        inputs = np.array(inputs)
        inputs = np.reshape(inputs, (self.numInputs, 1))

        hidden = np.matmul(self.weights_ih, inputs) + self.biases_h
        hidden = sigmoid_v(hidden)

        output = np.matmul(self.weights_ho, hidden) + self.biases_o
        output = sigmoid_v(output)

        targets = np.array(targets)
        targets = np.reshape(targets, (self.numOutputs, 1))

        output_errors = targets - output

        func = np.vectorize(lambda x : 1 - x)

        # Calculate deltas for weights and biases between hidden and output layer
        gradients_o = output * func(output)
        gradients_o = gradients_o * output_errors
        gradients_o = gradients_o * self.learning_rate

        delta_weights_ho = np.matmul(gradients_o, np.transpose(hidden))
        self.weights_ho += delta_weights_ho
        self.biases_o += gradients_o

        hidden_errors = np.matmul(np.transpose(self.weights_ho), output_errors)

        # Calculate deltas for weights and biases between input and hidden layer
        gradients_h = hidden * func(hidden)
        gradients_h = gradients_h * hidden_errors
        gradients_h = gradients_h * self.learning_rate

        delta_weights_ih = np.matmul(gradients_h, np.transpose(inputs))
        self.weights_ih += delta_weights_ih
        self.biases_h += gradients_h
