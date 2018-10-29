import numpy as np
import matplotlib.pyplot as plt
import math
import time

t = time.time()

### Niezbedne funkcje

def sigma(x):
    return 1 / (1 + math.exp(-x))

def sigma_derivative(x):
    return sigma(x) * (1 - sigma(x))

def neural_network(input, weight_1, weight_2):

    inp_data = np.array(input)
    w1 = np.array(weight_1)
    w2 = np.array(weight_2)

    ### Creation of neural network

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

    ### Training of neural network

    c = 0.1 # Współczynnik uczenia maszynowego
    n_n_output_0 = np.array(neural_network_output[0]) # matrix_mult_1
    n_n_output_1 = np.array(neural_network_output[1]) # matrix_mult_2
    n_n_output_2 = np.array(neural_network_output[2]) # layer_neuron_1
    n_n_output_3 = np.array(neural_network_output[3]) # layer_neuron_2
    input_data = np.array(neural_network_output[4])  # input data

    if (input_data[0][0] > 0) and (input_data[1][0] < 0):
        desired_output = [[1], [0]]  # 4 ćwiartka
    else:
        desired_output = [[0], [1]]  # każda inna

    # New weight number 2
    mistake_2 = desired_output - n_n_output_3
    delta_2 = [[] for i in range(len(n_n_output_1))]
    for i in range(len(n_n_output_1)):
        delta_2[i].append(np.asscalar(mistake_2[i] * sigma_derivative(n_n_output_1[i])))
    delta_2 = np.array(delta_2)
    new_weight_2 = weight_2 + c*delta_2 * np.transpose(n_n_output_2)
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

N = 100000# Wielkość próby

### Training of neural network

# Tworzenie próbki
sample = []
for i in range(N):
    input_data = np.random.uniform(-1, 1, (2, 1))
    false_output = [-1]
    connection = np.vstack([input_data, false_output])
    sample.append(connection)

# Procedura uczenia

fourth_right_classification = 0 # Liczba poprawnych klasyfikacji w 4 cwiartce
fourth_quarter_counter = 0 # Ogólna liczba punktow z 4 cwiartki
fourth_prob_array = [] # Tablica do przechowywania kolejnych prawdopodobieństw poprawnej klasyfikacji

other_right_classification = 0 # Liczba poprawnych klasyfikacji w pozostalych
other_quarter_counter = 0 # Ogólna liczba punktow z pozostalych cwiartek
other_prob_array = [] # Tablica do przechowywania kolejnych prawdopodobieństw poprawnej klasyfikacji

for i in range(N):

    output = neural_network(sample[i], weight_1, weight_2)
    new_weight_1, new_weight_2 = neural_learning(weight_1, weight_2, output)
    weight_1 = new_weight_1
    weight_2 = new_weight_2

    if (sample[i][0][0] > 0) and (sample[i][1][0] < 0):

        fourth_quarter_counter = fourth_quarter_counter + 1

        if output[3][0] > output[3][1]:
            fourth_right_classification = fourth_right_classification + 1
        else:
            pass

    else:

        other_quarter_counter = other_quarter_counter + 1

        if output[3][0] < output[3][1]:
            other_right_classification = other_right_classification + 1
        else:
            pass

    if fourth_quarter_counter != 0: # Dopóki żadna wartość z sample nie wpadnie do 4 ćwiartki to prawdp. jej poprawnej klasyfikacji ustawiamy na zerowe
        fourth_prob_array.append(fourth_right_classification / fourth_quarter_counter)
    else:
        fourth_prob_array.append(0)

    if other_quarter_counter != 0: # Dopóki żadna wartość z sample nie wpadnie do pozostałych ćwiartek to prawdp. ich poprawnej klasyfikacji ustawiamy na zerowe
        other_prob_array.append(other_right_classification / other_quarter_counter)
    else:
        other_prob_array.append(0)

right_prob_fourth = fourth_right_classification/fourth_quarter_counter
right_prob_other = other_right_classification/other_quarter_counter

elapsed = time.time() - t

### Prezentacja wyników

print("\nAmount of right classification in 4th quarter: ", fourth_right_classification, sep='')
print("Probability of right classification in 4th quarter after ", fourth_quarter_counter," samples as an input data: ", right_prob_fourth, sep='')
print("\nAmount of right classification in other quarters: ", other_right_classification, sep='')
print("Probability of right classification in other quarters after ", other_quarter_counter," samples as an input data: ", right_prob_other, sep='')
print("\nTime needed for the whole execution: ", round(elapsed, 2), "s", sep='')

plt.plot(fourth_prob_array, 'b')
plt.title('Efficiency of the learning procedure')
plt.ylabel('Probability of right classification')
plt.xlabel('Amount of samples as an input data')
plt.show()

### Test

# test_matrix = [[0.25],
#                [-0.7],
#                [-1]]
#
# test_output = neural_network(test_matrix, new_weight_1, new_weight_2)
#
# print("\nTested matrix: \n", test_matrix)
# print("\nTested output: \n", test_output[3])
#
# if (test_matrix[0][0] > 0) and (test_matrix[1][0] < 0):
#
#     if test_output[3][0] > test_output[3][1]:
#         print("\nIt's fourth quarter !")
#     else:
#         print("\nIt's not fourth quarter !")
#
# else:
#
#     if test_output[3][0] < test_output[3][1]:
#         print("\nIt's not fourth quarter !")
#     else:
#         print("\nIt's fourth quarter !")