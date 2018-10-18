import numpy as np
import math
import time

t = time.time()

def sigma(x):
    return 1 / (1 + math.exp(-x))

def sigma_derivative(x):
    return sigma(x) * (1 - sigma(x))

def neural_network(input, weight_1, weight_2):

    inp_data = np.array(input)
    w1 = np.array(weight_1)
    w2 = np.array(weight_2)

    ### Neural network

    # Layer neuron 1
    matrix_mult_1 = np.matmul(w1, inp_data)
    layer_neuron_1 = [[] for i in range(len(matrix_mult_1) + 1)]
    for i in range(len(matrix_mult_1)):
        layer_neuron_1[i].append(sigma(matrix_mult_1[i]))
    layer_neuron_1[-1].append(-1) # Dodajemy False Output zeby usunąc to b
    layer_neuron_1 = np.array(layer_neuron_1)

    # Layer neuron 2
    matrix_mult_2 = np.matmul(w2, layer_neuron_1)
    layer_neuron_2 = [[] for i in range(len(matrix_mult_2))]
    for i in range(len(matrix_mult_2)):
        layer_neuron_2[i].append(sigma(matrix_mult_2[i]))
    layer_neuron_2 = np.array(layer_neuron_2)

    return matrix_mult_1, matrix_mult_2, layer_neuron_1, layer_neuron_2, inp_data


def neural_learning(weight_1, weight_2, neural_network_output):

    ### Machine Learning

    desired_output = [[1], [0]]
    c = 0.1 # Współczynnik uczenia maszynowego
    n_n_output_0 = np.array(neural_network_output[0]) # matrix_mult_1
    n_n_output_1 = np.array(neural_network_output[1]) # matrix_mult_2
    n_n_output_2 = np.array(neural_network_output[2]) # layer_neuron_1
    n_n_output_3 = np.array(neural_network_output[3]) # layer_neuron_2
    input_data = np.array(neural_network_output[4])  # input data

    # New weight number 2
    mistake_2 = desired_output - n_n_output_3
    delta_2 = [[] for i in range(len(n_n_output_1))]
    for i in range(len(n_n_output_1)):
        delta_2[i].append(np.asscalar(mistake_2[i] * sigma_derivative(n_n_output_1[i])))
    delta_2 = np.array(delta_2)
    new_weight_2 = weight_2 + c *delta_2 * np.transpose(n_n_output_2)
    new_weight_2 = np.array(new_weight_2)

    # New weight number 1
    mistake_1 = np.matmul(np.transpose(weight_2), delta_2) # Ostatnia współrzędna jest zbędna, odpowiada fałszywemu wejściu -1 (które koduje poziom aktywacji), a więc pomijamy ją
    delta_1 = [[] for i in range(len(n_n_output_0))]
    for i in range(len(n_n_output_0)):
        delta_1[i].append(np.asscalar(mistake_1[i] * sigma_derivative(n_n_output_0[i])))
    delta_1 = np.array(delta_1)
    new_weight_1 = weight_1 + c *delta_1 * np.transpose(input_data)
    new_weight_1 = np.array(new_weight_1)

    return new_weight_1, new_weight_2

#### Parametry

weight_1 = [[0.1, -0.2, 0.3],
            [-0.4, 0.5, -0.6]]

weight_2 = [[0.15, -0.25, 0.35],
            [-0.45, 0.55, -0.65]]

input_data = [[1],
              [0],
              [-1]]

epsilon = 0.005 # Zadana dokładność jaką chcemy uzyskać

### Prezentacja wyników

output = neural_network(input_data, weight_1, weight_2)
print("\nOutput at the beginning:")
print(output[3])

counter = 0
while (abs(output[3][0] - 1) > epsilon) and (abs(output[3][1] - 0) > epsilon):
    new_weight_1, new_weight_2 = neural_learning(weight_1, weight_2, output)
    output = neural_network(input_data, new_weight_1, new_weight_2)
    weight_1 = new_weight_1
    weight_2 = new_weight_2
    counter = counter + 1
elapsed = time.time() - t

print("\nFinal output:")
print(output[3])
print("\nNumber of iterations: ", counter, sep='')
print("Time needed for the whole execution: ", round(elapsed, 2), "s", sep='')