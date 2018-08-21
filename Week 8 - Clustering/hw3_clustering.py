import numpy as np
import sys


def em_gmm(data, k_clusters):
    max_iteration = 10
    length = data.shape[0]
    dimensions = data.shape[1]
    sigma_k = np.eye(dimensions)

    # Initialize sigma with an identity matrix
    sigma = np.repeat(sigma_k[:, :, np.newaxis], k_clusters, axis=2)

    # Initialize a uniform probability distribution
    pi_class = np.ones(k_clusters) * (1 / k_clusters)
    phi = np.zeros((length, k_clusters))
    phi_normalized = np.zeros((length, k_clusters))

    # Initialize myu with random data points as in k_means
    indices = np.random.randint(0, length, size=k_clusters)
    myu = data[indices]

    for iteration in range(max_iteration):
        # Calculate Expectation state of EM Algorithm
        for k in range(k_clusters):
            inv_sigma_k = np.linalg.inv(sigma[:, :, k])
            inv_sqr_sigma_k_det = (np.linalg.det(sigma[:, :, k])) ** -0.5
            for index in range(length):
                xi = data[index, :]
                temp1 = (((xi - myu[k]).T).dot(inv_sigma_k)).dot(xi - myu[k])
                phi[index, k] = pi_class[k] * ((2 * np.pi) ** (-dimensions / 2)) * inv_sqr_sigma_k_det * np.exp(
                    -0.5 * temp1)
            for index in range(length):
                total = phi[index, :].sum()
                phi_normalized[index, :] = phi[index, :] / float(total)

        # compute maximization step of EM algorithm
        nK = np.sum(phi_normalized, axis=0)
        pi_class = nK / float(length)
        for k in range(k_clusters):
            myu[k] = ((phi_normalized[:, k].T).dot(data)) / nK[k]
        for k in range(k_clusters):
            # A column matrix containing zeros
            zc_matrix = np.zeros((dimensions, 1))
            # A dim*dim matrix containing zeros
            dim_sqr_matrix = np.zeros((dimensions, dimensions))
            for index in range(length):
                xi = data[index, :]
                zc_matrix[:, 0] = xi - myu[k]
                dim_sqr_matrix = dim_sqr_matrix + phi_normalized[index, k] * np.outer(zc_matrix, zc_matrix)
            sigma[:, :, k] = dim_sqr_matrix / float(nK[k])

        # Write outputs to files

        filename = "pi-" + str(iteration + 1) + ".csv"
        np.savetxt(filename, pi_class, delimiter=",")
        filename = "mu-" + str(iteration + 1) + ".csv"
        np.savetxt(filename, myu, delimiter=",")  # this must be done at every

        for j in range(k_clusters):  # k is the number of clusters
            # this must be done 5 times (or the number of clusters) for each iteration
            filename = "Sigma-" + str(j + 1) + "-" + str(iteration + 1) + ".csv"
            np.savetxt(filename, sigma[:, :, j], delimiter=",")


def k_means(data, k_clusters):
    # For now, taking 5 clusters and 10 iterations

    max_iteration = 10
    length = data.shape[0]
    # Cluster Assignment Vector
    cav = np.zeros(length)

    # Select random data points and initialize myu
    indices = np.random.randint(0, length, size=k_clusters)
    myu = data[indices]

    for iteration in range(max_iteration):
        # Update cluster assignments ci
        for i, xi in enumerate(data):
            temp_var = np.linalg.norm(myu - xi, 2, 1)
            cav[i] = np.argmin(temp_var)

        # Update the value of myu for the cluster
        n = np.bincount(cav.astype(np.int64), None, k_clusters)
        for k in range(k_clusters):
            indices = np.where(cav == k)[0]
            myu[k] = (np.sum(data[indices], 0)) / float(n[k])

        filename = "centroids-" + str(iteration + 1) + ".csv"  # "i" would be each iteration
        np.savetxt(filename, myu, delimiter=",")


def build_cluster():
    X = np.genfromtxt(sys.argv[1], delimiter=",")
    k_clusters = 5

    k_means(X, k_clusters)
    em_gmm(X, k_clusters)


if __name__ == '__main__':
    build_cluster()
