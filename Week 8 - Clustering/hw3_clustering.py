import numpy as np
import sys
from scipy.stats import multivariate_normal


# Expectation Maximization for Gaussian Mixture Model (EM GMM)
def em_gmm(data, k_clusters):
    max_iterations = 10

    rows = data.shape[0]
    columns = data.shape[1]

    # Initialize random centroids for clusters
    myu = data[np.random.choice(rows, k_clusters, replace=False), :]
    calc_sigma = [np.eye(columns) for i in range(k_clusters)]

    # Initializing Apriori distribution for pi
    pi = np.ones(k_clusters) / k_clusters

    phi = np.zeros((rows, k_clusters))
    for iteration in range(max_iterations):
        for i in range(rows):
            normalization_constant = np.sum(
                [pi[k] * multivariate_normal.pdf(data[i], mean=myu[k], cov=calc_sigma[k], allow_singular=True) for k in
                 range(k_clusters)])
            if normalization_constant == 0:
                phi[i] = pi / k_clusters
            else:
                phi[i] = [(pi[k] * multivariate_normal.pdf(data[i], mean=myu[k], cov=calc_sigma[k],
                                                           allow_singular=True)) / normalization_constant for k in
                          range(k_clusters)]

        for k in range(k_clusters):
            n_k = np.sum(phi[:, k])
            pi[k] = n_k / rows
            if n_k == 0:
                myu[k] = data[np.random.choice(rows, 1, replace=False), :]
                calc_sigma[k] = np.eye(columns)
            else:
                myu[k] = np.sum(data * phi[:, k].reshape(rows, 1), axis=0) / n_k
                covariance_sum = np.zeros((columns, columns))

                for i in range(rows):
                    centered_data = data[i] - myu[k]
                    covariance_sum += phi[i, k] * np.outer(centered_data, centered_data)

                calc_sigma[k] = covariance_sum / n_k

        filename = "pi-" + str(iteration + 1) + ".csv"
        np.savetxt(filename, pi, delimiter=",")
        filename = "mu-" + str(iteration + 1) + ".csv"
        np.savetxt(filename, myu, delimiter=",")  # this must be done at every iteration

        for j in range(k_clusters):  # k is the number of clusters
            filename = "Sigma-" + str(j + 1) + "-" + str(
                iteration + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, calc_sigma[j], delimiter=",")


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
