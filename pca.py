import sys
import numpy as np
import pandas as pd

"""
Train a PCA on the m=19 range finders and reduce to a small dimension l (=4)
to extract features and save weights to a npy file

This script runs as part of training.py

"""


class PCA():
    #
    # def __init__(self):
    #     self.m = 19
    #     self.l = 1
    #     self.weights = np.ones([self.m, self.l])
    #

    def getWeights():
        return np.load('./trained_nn/pca.npy')

    def convert(x):
        """
        Converts the range input data x into the pca out put
            - x must be in the shape (1, m) (for a single data sample)
              or in the shape (n, m) (for n data samples)
        """
        weights = np.load('./trained_nn/pca.npy')
        return np.dot(x, weights)

    # def load():
    #     """
    #     Load a previously saved set of weights
    #     """
    #     self.weights = np.load('./trained_nn/pca.npy')
    #     print(self.weights)

    def train(path_to_data):
        """
        Train the Principle Component Analysis and save the resulting weights
        """

        # number of inputs:
        m = 19
        # dimension to reduce to:
        l = 4
        # learning rate:
        lr_0 = 1e-3
        # number of iterations:
        iterations = 20000

        # positions and hence distances between input neurons:
        # angles = [-90.0, -75.0, -60.0, -45.0, -30.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]

        data_file = pd.read_csv(path_to_data, index_col=False)
        data = data_file.iloc[:, 6:]

        # Initialise weights:
        w = np.random.randn(m, l)

        # define a spreading distance for activation
        sig_0 = 0.1

        # distance function:
        def dist(i, j):
            return np.abs(i - j)

        while(True):
        # for n in range(iterations):
            # sample input:  (choose random!)
            x = np.array(data.iloc[np.random.randint(data.shape[0])])
            # x = np.random
            x.shape = (m, 1)

            # find winning neuron i:
            y = np.dot(w.T, x)
            i = np.argmax(y)

            h = np.zeros(l)

            # sig = sig_0 * np.exp(-n / 10)

            for j in range(l):
                h[j] = np.exp(-dist(i, j) / (2 * sig_0**2))

            delta_w = np.zeros([m, l])

            # lr = lr_0 * np.exp(-n / 10)

            for j in range(l):
                for i in range(m):
                    delta_w[i, j] = lr_0 * h[j] * (x[i] - w[i, j])

            w += delta_w

            if np.amax(delta_w) < 1e-6:
                print(w[0, :])
                break

            print(w[0, :])


            # Hebbian learning:
            # compute output:
            # y = np.dot(w, x)
            #
            # # change weights:
            # delta_w = np.zeros([l, m])
            #
            # for j in range(l):
            #     for i in range(m):
            #         sigma = 0
            #         for k in range(j):
            #             sigma += w[k][i] * y[k]
            #
            #         delta_w[j][i] = y[j] * (x[i] - sigma)
            #
            # # update weights:
            # w += (lr * delta_w)
            #
            # if (t+1)%1000 == 0:
            #     print(w[:, 0])

        weights = w
        np.save('./trained_nn/pca.npy', w)
