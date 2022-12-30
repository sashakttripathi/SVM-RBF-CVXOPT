import numpy as np
import cvxopt.solvers

import data as dt

x, y = dt.get_sample()
size = x.shape[0]

# distance parameter inside RBF
gamma = 0.1

# Kernel Matrix of RBF Kernel
K = np.array([np.exp(-gamma * (np.linalg.norm(x[i] - x[j]) ** 2))
              for j in range(size)
              for i in range(size)]).reshape(size, size)

# function to be minimized parameters
P = cvxopt.matrix(np.outer(y, y) * K)
q = cvxopt.matrix(-1 * np.ones(size))

# inequality constraint
G = cvxopt.matrix(np.diag(-1 * np.ones(size)))
h = cvxopt.matrix(np.zeros(size))

# equality constraint
A = cvxopt.matrix(y, (1, size))
b = cvxopt.matrix(0.0)

# solution
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# all alphas
alphas = np.ravel(solution['x'])

# only positive alphas' will contribute as support vectors  --> Core
positive_alpha_bool = alphas > 1e-7

# support vectors
support_vectors_x = x[positive_alpha_bool]
support_vectors_y = y[positive_alpha_bool]

# support vectors' alphas
support_alphas = alphas[positive_alpha_bool]


def compute_bias(x_support_vectors, y_support_vectors, support_v_alphas):
    bias_negative = np.sum([support_v_alphas[i] * y_support_vectors[i] *
                            np.exp(-gamma * (np.linalg.norm(x_support_vectors[i] - x_support_vectors[j]) ** 2))
                            for j in range(x_support_vectors.shape[0])
                            for i in range(x_support_vectors.shape[0])])
    return -1 * bias_negative / (x_support_vectors.shape[0] ** 2)


# compute the bias for our model
bias = compute_bias(support_vectors_x, support_vectors_y, support_alphas)
print("bias", bias)


def predict(x_test, y_test, x_support_vectors, y_support_vectors, support_v_alphas):
    correct = 0
    neg_but_given_pos = 0
    pos_but_given_neg = 0
    for i in range(x_test.shape[0]):
        x_i = x_test[i]
        predicted_value = np.sum([support_v_alphas[i] * y_support_vectors[i] *
                                  np.exp(-gamma * (np.linalg.norm(x_support_vectors[i] - x_i) ** 2))
                                  for i in range(x_support_vectors.shape[0])])

        if predicted_value >= 0:
            if y_test[i] > 0:
                correct += 1
            else:
                neg_but_given_pos += 1
            print(x_test[i], y_test[i], "prediction: ", 1)
        else:
            if y_test[i] < 0:
                correct += 1
            else:
                pos_but_given_neg += 1
            print(x_test[i], y_test[i], "prediction: ", -1)

    accuracy = (correct / x_test.shape[0]) * 100
    print(f"accuracy: {accuracy}")


predict(x, y, support_vectors_x, support_vectors_y, support_alphas)