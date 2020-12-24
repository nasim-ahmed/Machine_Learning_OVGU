import numpy as np
import argparse
import csv
from pathlib import Path


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#derivative of sigmoid activation function
def sigmoid_derivative(z):
    return z * (1 - z)


#initiliaze weights of neural network
def initialize_weights():
    np.random.seed(1)
    weights_hidden = np.array([[-0.30000, -0.10000, 0.20000], [0.40000, -0.40000, 0.1000]])

    bias_hidden = np.array([0.20000, -0.50000, 0.30000])

    weights_output = np.array([[0.10000, 0.30000, -0.40000]])


    weights_output = weights_output.reshape(-1, 1)
    bias_output = -0.10000

    return  weights_hidden, bias_hidden, weights_output, bias_output

#flatten a list of lists
def flatten_list(list_of_lists):
    flattened = []
    for sublist in list_of_lists:
        for val in sublist:
            flattened.append(val)
    return flattened


def write_to_csv(outputs):
    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")

        writer.writerow(["a","b","h1","h2","h3","o","t","delta_h1","delta_h2","delta_h3","delta_o","w_bias_h1","w_a_h1","w_b_h1","w_bias_h2",
                         "w_a_h2","w_b_h2","w_bias_h3","w_a_h3","w_b_h3","w_bias_o","w_h1_o","w_h2_o","w_h3_o"])
        writer.writerow(["-","-","-","-","-","-","-","-","-","-","-",   0.20000,  -0.30000,   0.40000,  -0.50000,  -0.10000,  -0.40000,   0.30000,   0.20000,   0.10000,  -0.10000,   0.10000,   0.30000,  -0.40000])

        writer.writerows(outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '--data', help='Path+Name of file')
    parser.add_argument('--eta', '--eta')
    parser.add_argument('--iterations', '--iterations')

    args = parser.parse_args()
    return args


def read_csv(data):
    with open(data, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)

    X = data[:, :-1]
    Y = data[:, -1]
    Y = Y.reshape(-1,1)

    return X, Y


if __name__ == '__main__':
    w_hidden, b_hidden, w_output, b_output = initialize_weights()

    args = parse_args()

    feature_set, labels = read_csv(args.data)

    out_file_name = Path(args.data).stem+'_Solution'

    no_of_iterations = int(args.iterations)
    lr = float(args.eta)

    header2 = ["-","-","-","-","-","-","-","-","-","-","-",   0.20000,  -0.30000,   0.40000,  -0.50000,  -0.10000,  -0.40000,   0.30000,   0.20000,   0.10000,  -0.10000,   0.10000,   0.30000,  -0.40000]
    print(*header2, sep=',   ')

    for epoch in range(0,no_of_iterations):
        for X, Y in zip(feature_set, labels):
            local_list = []

            local_list.append(list(X))

            X = X.reshape(1, -1).astype(float)

            H = sigmoid(np.dot(X, w_hidden) + b_hidden)
            local_list.append(flatten_list(H.tolist()))

            O = sigmoid(np.dot(H, w_output) + b_output)

            local_list.append(flatten_list(O))
            local_list.append(list(Y))

            # Error gradient for single output unit
            delta_O = (Y - O) * sigmoid_derivative(O)

            # Error gradient for each hidden unit
            delta_H = sigmoid_derivative(H) * (w_output.T * delta_O)

            local_list.append(flatten_list(delta_H))

            local_list.append(flatten_list(delta_O))

            # weight update for hidden layer
            b_hidden = b_hidden + (lr * delta_H * 1)

            w_hidden = w_hidden + (lr * X.T.dot(delta_H))

            for i in range(0, b_hidden.shape[1]):
                local_list.append(flatten_list(b_hidden[:, i:i + 1]))
                local_list.append(flatten_list(w_hidden[:, i:i + 1]))

            # weight update for output layer
            b_output += (lr * delta_O * 1)
            local_list.append(flatten_list(b_output))
            w_output += (lr * H.T.dot(delta_O))

            local_list.append(flatten_list(w_output))

            print(",   ".join(repr(round(e, 5)) for e in flatten_list(local_list)))
            local_list.clear()

















