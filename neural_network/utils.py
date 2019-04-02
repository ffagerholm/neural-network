"""
Activation and loss functions.
Author: Fredrik Fagerholm
"""
import numpy as np


# activation functions and their derivatives
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(x):
    return np.maximum(np.zeros(x.shape), x)

def relu_prime(x):
    return np.where(x > 0, np.ones(x.shape), np.zeros(x.shape))

def identity(x):
    return x

def identity_prime(x):
    return np.ones(x.shape)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def softmax_prime(x):
    s = softmax(x)
    return np.multiply(s, 1 - s)

activation_functions = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'relu': (relu, relu_prime),
    'softmax': (softmax, softmax_prime),
    'identity': (identity, identity_prime),
}


# cost functions and their derivatives
def squared_error(y_true, y_pred):
    return (y_true - y_pred)**2
    
def squared_error_prime(y_true, y_pred):
    return 2*(y_true - y_pred)

def cross_entropy(y_true, y_pred):
    return np.sum(np.nan_to_num(-y_true * np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred)))

def cross_entropy_prime(y_true, y_pred):
    return y_true - y_pred

loss_functions = {
    'squared_error': (squared_error, squared_error_prime),
    'cross_entropy': (cross_entropy, cross_entropy_prime),
}