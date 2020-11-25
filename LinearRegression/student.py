##################################################################################################;
#
# author            : Nasim
# date              : 24.11.2020
# assignment        : Programming Assignment 1
# filename          : student.py
# description       : linear regression implementation using
#                     batch gradient descent
#
##################################################################################################;

import argparse
import numpy as np
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '--data', help='Path+Name of file')
    parser.add_argument('--eta', '--eta')
    parser.add_argument('--threshold', '--threshold')

    args = parser.parse_args()
    return args

def linear_regression(data, eta, threshold):
    with open(data, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)
        num_rows = data.shape[0]
        data = np.insert(data, 0, np.ones(num_rows), axis=1)

    X = data[:, :-1]
    Y = data[:, -1]

    W = np.zeros(X.shape[1]).astype(float)

    prev_err = np.sum(np.square(Y - np.matmul(X, W))).astype(float)
    step = 1

    print('{},'.format(step), end="")
    for w in W:
        print('{0:.9f},'.format(w), end="")

    print('{0:.9f}'.format(prev_err))

    while True:
        step += 1

        grad = np.matmul(X.transpose(), (Y - np.matmul(X, W)))

        W = W + (np.array(eta).astype(np.float) * grad)

        pres_err = np.sum(np.square(Y - np.matmul(X, W))).astype(float)
        print('{},'.format(step), end="")
        for w in W:
            print('{0:.9f},'.format(w), end="")

        print('{0:.9f}'.format(pres_err))
        # Threshold check
        if  prev_err - pres_err < np.array(threshold).astype(np.float):
            break

        prev_err = pres_err


if __name__ == '__main__':
    args = parse_args()


    linear_regression(args.data, args.eta, args.threshold)



