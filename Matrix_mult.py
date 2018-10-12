import numpy as np

def multiply(a, b):
    try:
        result = [[0 for col in range(len(max(b, key=len)))] for row in range(len(a))]
        for r in range(len(a)):
            for c in range(len(max(b, key=len))):
                for k in range(len(b)):
                    result[r][c] += a[r][k] * b[k][c]
        return result
    except:
        print("Exception! Matrices in 'multiply' function are wrong defined!")

def multiplyWithNumpy(a, b):
    try:
        x = np.array(a)
        y = np.array(b)
        result = np.matmul(x, y)
        return result
    except:
        print("Exception! Matrices in 'multiplyWithNumpy' function are wrong defined!")


# Example of good matrices
X = [[12, 7, 3],
     [4, 5, 6],
     [7, 8, 9]]

Y = [[5, 8, 1, 2],
     [6, 7, 3, 0],
     [4, 5, 9, 1]]

# Example of bad matrices
A = [[1, -2, 3],
     [-3, -2, 1]]

B = [[1, -2],
     [3, 4],
     [-1, -1, 1]]

# Case with X, Y matrices is presented
result = multiply(X, Y)
if np.any(result) != None:
    print("Result obtained without using numphy module:\n", result)

result_numpy = multiplyWithNumpy(X, Y)
if np.any(result_numpy) != None:
    print("Result obtained by using numphy module:\n", result_numpy)
