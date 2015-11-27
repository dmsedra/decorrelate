#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import sklearn.datasets
from layers import *
from helpers import *
import neuralnetwork as net


def run():
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    split = 10000

    perm = np.random.permutation(mnist.data.shape[0])
    trainIdxs = perm[:split]
    testIdxs = perm[split:]
    X_train = mnist.data[trainIdxs]/255.0
    y_train = mnist.target[trainIdxs]
    X_test = mnist.data[testIdxs]/255.0
    y_test = mnist.target[testIdxs]
    n_classes = np.unique(y_train).size

    print X_train.shape, np.amax(X_train)

    # Setup multi-layer perceptron
    nn = net.NeuralNetwork(
        layers=[
            Linear(
                n_out=500,
                weight_scale=0.2,
                weight_decay=0.008,
            ),
            Activation('relu'),
            Whiten(
                n_out=200,
                weight_scale=0.2,
                weight_decay=0.008,
            ),
            Activation('relu'),
            Linear(
                n_out=n_classes,
                weight_scale=0.2,
                weight_decay=0.004,
            ),
            LogRegression(),
        ]
   )

    #nn.check_gradients(X_train[:50],y_train[:50])
    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.25, max_iter=10, batch_size=1000)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
