import numpy as np
import sys


## Solution for Part 1
def part1(lambda_input, sigma2_input, X_train, y_train):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    return 0



## Solution for Part 2
def part2(lambda_input, sigma2_input, X_train, y_train, X_test):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    return 0



if __name__ == '__main__':
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter=",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter=",")

    # Compute wRR from the first part
    wRR = part1(lambda_input, sigma2_input, X_train, y_train)
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")  # write output to file

    # Compute sequence for Active Learning from the second part
    active = part2(lambda_input, sigma2_input, X_train, y_train, X_test.copy())  # Assuming active is returned from the function
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active,
               delimiter=",")  # write output to file