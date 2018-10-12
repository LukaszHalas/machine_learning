import numpy as np
import matplotlib.pyplot as plt
import math

def sigma(x):
    return 1 / (1 + math.exp(-x))

def neural_network(input, weight_1, weight_2):

    inp_data = np.array(input)
    w1 = np.array(weight_1)
    w2 = np.array(weight_2)
    layer_neuron_1 = [[],
                      [],
                      []]
    layer_neuron_2 = [[],
                      []]

    # Layer neuron 1
    matrix_mult_1 = np.matmul(w1, inp_data)
    for i in range(len(matrix_mult_1)):
        layer_neuron_1[i].append(sigma(matrix_mult_1[i]))
    layer_neuron_1[-1].append(-1) # Dodajemy False Output

    # Layer neuron 2
    matrix_mult_2 = np.matmul(w2, layer_neuron_1)
    for i in range(len(matrix_mult_2)):
        layer_neuron_2[i].append(sigma(matrix_mult_2[i]))

    # # outputLayer1 = sigma( firstLayerNeuron1 -  firstLayerNeuron2 + 0.3)
    # # outputLayer2 = sigma(-  firstLayerNeuron1 + firstLayerNeuron2 - 0.3)
    return layer_neuron_2

# Parametry
weight_1 = [[0.1, -0.2, 0.3],
            [-0.4, 0.5, -0.6]]
weight_2 = [[0.15, -0.25, 0.35],
            [-0.45, 0.55, -0.65]]

input_data = [[1],
              [0],
              [-1]]

x = neural_network(input_data, weight_1, weight_2)
print(x)
