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
    X_train = mnist.data[:split]/255.0
    y_train = mnist.target[:split]
    X_test = mnist.data[split:]/255.0
    y_test = mnist.target[split:]
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 10000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]

    # Setup multi-layer perceptron
    nn = net.NeuralNetwork(
        layers=[
            Linear(
                n_out=100,
                weight_scale=0.2,
                weight_decay=0.004,
            ),
            Activation('relu'),
            Whiten(
                n_out=100,
                weight_scale=0.2,
                weight_decay=0.004,
            ),
            Activation('relu'),
            Linear(
                n_out=n_classes,
                weight_scale=0.2,
                weight_decay=0.004,
            ),
            LogRegression(),
        ],
    )

    nn.check_gradients(X_train[:50],y_train[:50])
    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.1, max_iter=20, batch_size=64)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)


if __name__ == '__main__':
    run()
