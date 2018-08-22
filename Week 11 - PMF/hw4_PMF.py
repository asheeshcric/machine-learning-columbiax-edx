from __future__ import division
import sys
import numpy as np
import pandas as panduram


# Implement function here
def PMF(train_data, lambda_parameter, variance, dimensions, max_iteration):
    # Initialize L matrix
    L = np.zeros(max_iteration)

    # Initialize U_matrices
    n_u = train_data.shape[0]
    U_matrices = np.zeros((max_iteration, n_u, dimensions))

    # Initialize V_matrices
    n_v = train_data.shape[1]
    V_matrices = np.zeros((max_iteration, n_v, dimensions))

    # Initialize and calculate mean and covariance
    myu = np.zeros(dimensions)
    covariance = (1 / lambda_parameter) * np.identity(dimensions)
    V_matrices[0] = np.random.multivariate_normal(myu, covariance, n_v)

    for iteration in range(max_iteration):
        length = 0 if iteration == 0 else iteration - 1
        for i in range(n_u):
            matrix1 = lambda_parameter * variance * np.identity(dimensions)
            matrix2 = np.zeros(dimensions)
            for j in range(n_v):
                if train_data[i, j] == True:
                    matrix1 += np.outer(V_matrices[length, j], V_matrices[length, j])  # movie rated by the user
                    matrix2 += train_data[i, j] * V_matrices[length, j]

            U_matrices[iteration, i] = np.dot(np.linalg.inv(matrix1), matrix2)

        for j in range(n_v):
            matrix1 = lambda_parameter * variance * np.identity(dimensions)
            matrix2 = np.zeros(dimensions)
            for i in range(n_u):
                if train_data[i, j] == True:
                    matrix1 += np.outer(U_matrices[iteration, i], U_matrices[iteration, i])
                    matrix2 += train_data[i, j] * U_matrices[iteration, i]

            V_matrices[iteration, j] = np.dot(np.linalg.inv(matrix1), matrix2)

        current_temp = 0
        for i in range(n_u):

            for j in range(n_v):

                if train_data[i, j] == True:
                    current_temp -= np.square(
                        train_data[i, j] - np.dot(U_matrices[iteration, i].T, V_matrices[iteration, j]))

        current_temp = (1 / (2 * variance)) * current_temp

        current_temp -= (lambda_parameter / 2) * (
                np.square(np.linalg.norm(U_matrices[iteration])) + np.square(np.linalg.norm(V_matrices[iteration])))

        L[iteration] = current_temp

    return L, U_matrices, V_matrices


def setup_pmf():
    data = panduram.read_csv(sys.argv[1])
    data = data.pivot(index=data.columns[0], columns=data.columns[1],
                      values=data.columns[2])
    unused_variable = ~data.isnull().as_matrix()
    train_data = data.as_matrix()

    lambda_parameter, variance, dimensions, max_iteration = 2, 0.1, 5, 50

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
    L, U_matrices, V_matrices = PMF(train_data, lambda_parameter, variance, dimensions, max_iteration)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


if __name__ == '__main__':
    setup_pmf()
