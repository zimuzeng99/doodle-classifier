import neural_network
import random
import numpy as np

car_data = np.load("car.npy")
airplane_data = np.load("airplane.npy")
aircraft_carrier_data = np.load("aircraft_carrier.npy")

nn = neural_network.NeuralNetwork(784, 64, 3)

def normalize(x):
    return x / 255.0

normalize_v = np.vectorize(normalize)

def index_max(output):
    index = 0
    max = 0

    if output[0][0] > max:
        max = output[0][0]
        index = 0

    if output[1][0] > max:
        max = output[1][0]
        index = 1

    if output[2][0] > max:
        max = output[2][0]
        index = 2

    return index


for i in range(0, 25000):
    nn.train(normalize_v(car_data[i]).tolist(), [1.0, 0.0, 0.0])
    nn.train(normalize_v(airplane_data[i]).tolist(), [0.0, 1.0, 0.0])
    nn.train(normalize_v(aircraft_carrier_data[i]).tolist(), [0.0, 0.0, 1.0])
    print(i)

numCorrect = 0
for i in range(30000, 32000):
    output = index_max(nn.feed_forward(normalize_v(car_data[i]).tolist()))
    if output == 0:
        numCorrect += 1

    output = index_max(nn.feed_forward(normalize_v(airplane_data[i]).tolist()))
    if output == 1:
        numCorrect += 1

    output = index_max(nn.feed_forward(normalize_v(aircraft_carrier_data[i]).tolist()))
    if output == 2:
        numCorrect += 1

print(numCorrect / 6000)
