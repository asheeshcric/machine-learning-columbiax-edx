import numpy as np
import sys


## Solution for Part 1
def part1(lambda_input, X_train, y_train):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    """
    Objective: We can solve the ridge regression problem using exactly the
    same procedure as for least squares,
    L = ky − Xwk 2 + λkwk 2 = (y − Xw) T (y − Xw) + λw T w.
    Solution: First, take the gradient of L with respect to w and set to zero,
        ∇ w L = −2X T y + 2X T Xw + 2λw = 0
    Then, solve for w to find that
        w RR = (λI + X T X) −1 X T y.
    """
    dimensions = X_train.shape[1]
    # Finding (λI + X T X)
    expression = lambda_input * np.eye(dimensions) + (X_train.T).dot(X_train)
    # Calculating wRR by inversing above expression and multiplying to X(T) and y
    final_wRR = (np.linalg.inv(expression)).dot((X_train.T).dot(y_train))
    return final_wRR


def update_posterior_distribution(lambda_input, sigma2_input, X_train, dimensions, y_test, auto_correlation,
                                  cross_correlation):
    auto_correlation = (X_train.T).dot(X_train) + auto_correlation
    cross_correlation = (X_train.T).dot(y_test) + cross_correlation

    covariance_inverse = lambda_input * np.eye(dimensions) + (1 / sigma2_input) * auto_correlation
    covariance = np.linalg.inv(covariance_inverse)

    temp1 = lambda_input * sigma2_input * np.eye(dimensions) + auto_correlation
    mean = (np.linalg.inv(temp1)).dot(cross_correlation)

    return covariance, mean, auto_correlation, cross_correlation


## Solution for Part 2
def part2(lambda_input, sigma2_input, X_train, y_train, X_test):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    dimensions = X_train.shape[1]
    active = []
    auto_correlation = np.zeros((dimensions, dimensions))
    cross_correlation = np.zeros(dimensions)
    covariance, wRR, auto_correlation, cross_correlation = update_posterior_distribution(lambda_input, sigma2_input,
                                                                                         X_train,
                                                                                         dimensions, X_test,
                                                                                         auto_correlation,
                                                                                         cross_correlation)
    # Here wRR is directly taken as mean from Lecture 5, slide 19
    label_indices = list(range(X_test.shape[0]))
    for i in range(10):
        variance_matrix = (X_test.dot(covariance)).dot(X_test.T)
        row = np.argmax(variance_matrix.diagonal())
        X_data = X_test[row, :]
        label = X_data.dot(wRR)  # From Lecture 5, slide 12
        # Append active rows
        active.append(label_indices[row])
        X_test = np.delete(X_test, (row), axis=0)
        label_indices.pop(row)

        # Once again, update the posterior distribution
        covariance, wRR, auto_correlation, cross_correlation = update_posterior_distribution(lambda_input, sigma2_input,
                                                                                             X_data,
                                                                                             dimensions, X_test,
                                                                                             auto_correlation,
                                                                                             cross_correlation)

    return [row + 1 for row in active]


if __name__ == '__main__':
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter=",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter=",")

    # Compute wRR from the first part
    wRR = part1(lambda_input, X_train, y_train)
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")  # write output to file

    # Compute sequence for Active Learning from the second part
    active = part2(lambda_input, sigma2_input, X_train, y_train,
                   X_test.copy())  # Assuming active is returned from the function
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active,
               delimiter=",")  # write output to file
