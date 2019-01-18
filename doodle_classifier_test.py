import neural_network
import random
import math
import numpy as np
from mnist import MNIST

nn = neural_network.NeuralNetwork([784, 400, 400, 3], 0.2)

car_data = np.load("car.npy")
airplane_data = np.load("airplane.npy")
aircraft_carrier_data = np.load("aircraft_carrier.npy")

def normalize(x):
    return x / 255

v_normalize = np.vectorize(normalize)

training_size = 25000

# Normalizes the training set to improve learning effectiveness
for i in range(0, training_size):
    car_data[i] = normalize(car_data[i])
    airplane_data[i] = normalize(airplane_data[i])
    aircraft_carrier_data[i] = normalize(aircraft_carrier_data[i])

num_iterations = 0
while num_iterations < 15:
    for i in range(0, training_size):
        nn.train(car_data[i].tolist(), [1, 0, 0])
        nn.train(airplane_data[i].tolist(), [0, 1, 0])
        nn.train(aircraft_carrier_data[i].tolist(), [0, 0, 1])
    num_iterations += 1

num_correct = 0

for i in range(0, training_size):
    # The output will be a vector of probabilities. We use the largest value
    # in this vector to determine the digit that the NN recognized.
    output = nn.compute(car_data[i])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 0:
        num_correct += 1

    output = nn.compute(airplane_data[i])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 1:
        num_correct += 1

    output = nn.compute(aircraft_carrier_data[i])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 2:
        num_correct += 1

accuracy = num_correct / (training_size * 3)
print("Training accuracy: " + str(accuracy))

test_size = 1000

num_correct = 0

for i in range(0, test_size):
    # The output will be a vector of probabilities. We use the largest value
    # in this vector to determine the digit that the NN recognized.
    output = nn.compute(car_data[i + 10000])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 0:
        num_correct += 1

    output = nn.compute(airplane_data[i + 10000])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 1:
        num_correct += 1

    output = nn.compute(aircraft_carrier_data[i + 10000])
    doodle_class = output.index(max(output))

    # If the NN predicted the correct digit
    if doodle_class == 2:
        num_correct += 1

accuracy = num_correct / (test_size * 3)
print("Test accuracy: " + str(accuracy))

np.save("weights0", nn.weights[0])
np.save("weights1", nn.weights[1])
np.save("weights2", nn.weights[2])

np.save("biases0", nn.biases[0])
np.save("biases1", nn.biases[1])
np.save("biases2", nn.biases[2])
