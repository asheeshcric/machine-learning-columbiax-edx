from __future__ import division
import numpy as np
import sys


# Probabilistic Matrix Factorization
def PMF(train_data):
    # Lecture 17, slide 19
    dimensions, myu, variance, lambda_parameter, max_iteration = 5, 0, 0.1, 2, 50
    L = np.zeros((max_iteration, 1))

    # Get dimension of matrix M
    n_u = int(np.amax(train_data[:, 0]))
    n_v = int(np.amax(train_data[:, 1]))

    V_matrices = np.random.normal(myu, np.sqrt(1 / lambda_parameter), (n_v, dimensions))
    U_matrices = np.zeros((n_u, dimensions))

    index_ui = []
    for i in range(n_u):
        mat1 = train_data[train_data[:, 0] == i + 1][:, 1]  # index set of objects rated by user i
        mat2 = mat1.astype(np.int64)
        index_ui.append(mat2)

        # Forming V_matrices
    index_vj = []
    for j in range(n_v):
        mat1 = train_data[train_data[:, 1] == j + 1][:, 0]  # index set of users who rated object j
        mat2 = mat1.astype(int)
        index_vj.append(mat2)

    # Forming M_matrices
    m_matrix = np.zeros((n_u, n_v))
    for val in train_data:
        row = int(val[0])
        col = int(val[1])
        m_matrix[row - 1, col - 1] = val[2]

    for iteration in range(max_iteration):
        # Update U_matrices
        for i in range(n_u):
            mat1 = lambda_parameter * variance * np.eye(dimensions)
            mat2 = V_matrices[index_ui[i] - 1]
            mat3 = (mat2.T).dot(mat2)
            mat4 = np.linalg.inv(mat1 + mat3)

            mat5 = m_matrix[i, index_ui[i] - 1]
            mat6 = (mat2 * mat5[:, None]).sum(axis=0)

            ui = mat4.dot(mat6)
            U_matrices[i] = ui

        # Update V_matrices
        for j in range(n_v):
            mat1 = lambda_parameter * variance * np.eye(dimensions)
            mat2 = U_matrices[index_vj[j] - 1]
            mat3 = (mat2.T).dot(mat2)
            mat4 = np.linalg.inv(mat1 + mat3)

            mat5 = m_matrix[index_vj[j] - 1, j]
            mat6 = (mat2 * mat5[:, None]).sum(axis=0)

            vj = mat4.dot(mat6)
            V_matrices[j] = vj

        # Calculate MAP objective functions
        map2 = lambda_parameter * 0.5 * (((np.linalg.norm(U_matrices, axis=1)) ** 2).sum())
        map3 = lambda_parameter * 0.5 * (((np.linalg.norm(V_matrices, axis=1)) ** 2).sum())
        map1 = 0
        for val in train_data:
            i = int(val[0])
            j = int(val[1])
            map1 = map1 + (val[2] - np.dot(U_matrices[i - 1, :], V_matrices[j - 1, :])) ** 2
        map1 = map1 / (2 * variance)
        L[iteration] = - map1 - map2 - map3

    return L, U_matrices, V_matrices


def setup_pmf():
    train_data = np.genfromtxt(sys.argv[1], delimiter=",")

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
    L, U_matrices, V_matrices = PMF(train_data)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


if __name__ == '__main__':
    setup_pmf()
