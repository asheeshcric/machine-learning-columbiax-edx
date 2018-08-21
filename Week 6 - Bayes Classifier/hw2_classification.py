from __future__ import division
import numpy as np
import sys


# can make more functions if required

def conditional_density(X_train, y_train, k_class):
    length = X_train.shape[0]
    dimensions = X_train.shape[1]
    cov = np.zeros((dimensions, dimensions, k_class))
    myu = np.zeros((k_class, dimensions))

    unique, counts = np.unique(y_train, return_counts=True)
    pi = (counts / float(length)).T

    for i in range(k_class):
        xi = X_train[(y_train == i)]
        myu[i] = np.mean(xi, axis=0)
        xi_normalized = xi - myu[i]
        temp_cov = (xi_normalized.T).dot(xi_normalized)
        cov[:, :, i] = temp_cov / float(len(xi))

    return pi, myu, cov


def plugin_classifier(X_test, pi, myu, cov, k_class):
    # this function returns the required output
    length = X_test.shape[0]
    prob = np.zeros((length, k_class))
    norm_probabs = np.zeros((length, k_class))

    for k in range(k_class):
        inv_cov = np.linalg.inv(cov[:, :, k])
        cov_inv_sqr_det = (np.linalg.det(cov[:, :, k])) ** -0.5
        for index in range(length):
            x0 = X_test[index, :]
            temp1 = (((x0 - myu[k]).T).dot(inv_cov)).dot(x0 - myu[k])
            prob[index, k] = pi[k] * cov_inv_sqr_det * np.exp(-0.5 * temp1)

    for index in range(length):
        total = prob[index, :].sum()
        norm_probabs[index, :] = prob[index, :] / float(total)
    return norm_probabs


def build_model():
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    k_class = 10

    pi, myu, cov = conditional_density(X_train, y_train, k_class)

    final_outputs = plugin_classifier(X_test, pi, myu, cov, k_class)  # assuming final_outputs is returned from function

    np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file


if __name__ == '__main__':
    build_model()
