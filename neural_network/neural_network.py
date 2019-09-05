# -*- coding: utf-8 -*-
"""
Implementation of a basic fully connected feedforward neural network.

Author: Fredrik Fagerholm
"""
import numpy as np
from .utils import activation_functions, loss_functions


class Layer:
    """Fully connected neural network layer.
    Args:
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        activation_function (str): Activation function applied to output.
            Can be one of 'sigmoid', 'relu', 'softmax' or 'identity'.
            Default is 'sigmoid'.
        layer_nr (int): Identifier for layer.
    """
    def __init__(self, in_size, out_size, activation_function='sigmoid', layer_nr=0):
        self.in_size = in_size
        self.out_size = out_size
        
        # for debugging
        self.layer_nr = layer_nr
        
        # initialize weights with normally distributed random values
        self.weights = np.random.normal(size=(self.in_size, self.out_size))
        # initialize bias vector with uniformly random values
        self.bias = np.random.random(size=(1, self.out_size))
        
        self.init_gradients()
        self.activation = np.zeros(shape=(self.out_size, 1))

        self.afn = activation_function
        self.af, self.afd = activation_functions[activation_function]
        
    def init_gradients(self):
        """Resets gradients.
        Sets layer gradients to zero.
        """
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        
    def compute_activation(self, x):
        """Computes layer activation.
        Compute output of layer.
        Args:
            x (numpy.array): Input. Should have dimensions (1, in_size)
        
        Returns:
            activation (numpy.array): (1, out_size) array containing the outputs.
        """
        if x.shape != (1, self.in_size):
            raise ValueError("Layer {} expected input of shape {} got input of shape {}".format(
                        self.layer_nr, (1, self.in_size), x.shape))
        
        # store input for gradient computation
        self.upstream_activation = x
        # compute the weighted input and store for gradient computation
        self.z = np.dot(x, self.weights) + self.bias
        # apply activation function
        self.activation = self.af(self.z)
        return self.activation
    
    def feedforward(self, x):
        """Compute output of layer.
        Alias for `compute_activation`.
        """
        return self.compute_activation(x)
    
    def predict(self, x, threshold=0.5):
        """Generates predicted class.
        Computes activation, if the shape if the activation is (1, 1)
        the layers predicts a binary classification problem: either 0 or 1
        depending on if the activation surpasses the given threshold (default 0.5).
        
        If the dimension of the activation vector is greater than 1, the layer predicts
        the class that corresponds to the position in the vector with the greatest 
        activation value.
        
        Args:
            x (numpy.array): Input. Should have dimensions (1, in_size)
            threshold (float): Threshold used in binary classification.
                Default value is 0.5.
        
        Returns:
            predicted_class (int)
        """
        activation = self.compute_activation(x)
        # if the activation is a (1, 1)-array there are only 2 classes,
        # otherwise it should be treated as a multinomial distribution
        if activation.shape[1] == 1:
            return activation > threshold
        else:
            return np.argmax(activation)
    
    def compute_gradient(self, output_grad):
        """Computes gradients for layer.

        Computes the gradients with respect to the weights and bias in 
        the layer, as well as with respect to the activation of the previous
        layers. The gradients for the this layer are stored in instance variables, 
        the activation gradient is returned.
        
        Args: 
            output_grad (numpy.array): Gradient w.r.t the output of this layer.
                Should have shape (1, out_size)
        
        Returns: 
            activation_grad (numpy.array): Gradient w.r.t the input of this layer. 
        """
        if output_grad.shape != (1, self.out_size):
            raise ValueError("Layer {} expected a gradient of shape {} got input of shape {}".format(
                        self.layer_nr, (1, self.out_size), output_grad.shape))
        
        # gradient is the elementwise product between the 
        # gradient of the activation function at z, and the loss gradient
        grad = np.multiply(self.afd(self.z), output_grad)
        # gradient w.r.t. the weights in this layer
        self.weights_gradient += np.outer(self.upstream_activation.T, grad)
        # gradient w.r.t. the bias in this layer
        self.bias_gradient += np.copy(grad)
        # gradient w.r.t. the activation from the previous layer
        activation_grad = np.dot(grad, self.weights.T)
        return activation_grad       
    
    def update_weights(self, train_size, lr=0.01, l2=0.0, batch_size=1):
        """Updates layer weights.
        Args: 
            lr (float): Learning rate. Used for controling how large the step in the gradient 
                decent update should be. Default is 0.01.
            batch_size (int): Number of examples in batch. Used to normalize the update to the 
                batch size.
            l2 (float): L2 regularization strength.
        """
        # L2 regularization (shrinkage) for weights
        self.weights *= (1 - lr*(l2/train_size))
        # Update weights and bias
        self.weights += (lr/batch_size)*self.weights_gradient 
        self.bias += (lr/batch_size) * self.bias_gradient
    
    def __repr__(self):
        return "Layer(in={:d}, out={:d}, activation_function={})".format(self.in_size, self.out_size, self.afn)


class NeuralNetwork:
    """Fully connected neural network.

    Args:
        layer_sizes (list(int)): List of layer sizes. The first value will be the size of the 
            input layer, the last value will be the size of the output layer. If the list has length 
            n the network will have n - 1 layers. 
        activation_functions (list(str)): List of activation functions to use in each layer. 
            Can be one of 'sigmoid', 'relu', 'softmax' or 'identity'.
        loss_function (str): Loss function to use in the network. Can be one
            of 'squared_error' or 'cross_entropy'. Default is 'squared_error'.
    """
    def __init__(self, layer_sizes, activation_functions=[], loss_function='squared_error'):
        self.num_layers = len(layer_sizes)
        if activation_functions:
            self.layers = [Layer(in_size, out_size, activation_function=afn, layer_nr=i) for i, (in_size, out_size, afn) in
                            enumerate(zip(layer_sizes, layer_sizes[1:], activation_functions))]
        else:
            self.layers = [Layer(in_size, out_size, layer_nr=i) for i, (in_size, out_size) in
                            enumerate(zip(layer_sizes, layer_sizes[1:]))]
            
        self.loss_function, self.loss_diff = loss_functions[loss_function]
        
    def feedforward(self, x):
        """Computes output of network.
        Args:
            x (numpy.array): Input. Should have dimensions (1, layer_sizes[0])
        Returns:
            activation (numpy.array): (1, layer_sizes[-1]) array containing the outputs.
        """
        activation = x.copy()
        for layer in self.layers:
           activation = layer.compute_activation(activation)
        return activation
    
    def predict(self, x, threshold=0.5):
        """Generates predicted class.
        Computes the output layer activation, if the shape if the activation is (1, 1)
        the layers predicts a binary classification problem: either 0 or 1
        depending on if the activation surpasses the given threshold (default 0.5).
        
        If the dimension of the activation vector is greater than 1, the layer predicts
        the class that corresponds to the position in the vector with the greatest 
        activation value.
        
        Args:
            x (numpy.array): Input. Should have dimensions (1, layer_sizes[0])
            threshold (float): Threshold used in binary classification.
                Default value is 0.5.
        
        Returns:
            predicted_class (int)
        """
        activation = self.feedforward(x)
        if activation.size == 1:
            # only one output
            return activation > threshold
        else:
            # output is a multinomial
            return np.argmax(activation)

    def backpropagate(self, gradient):
        """Backpropagates the gradient through the network.
        Args:
            gradient (numpy.array): Gradient w.r.t to the output layer.
        """
        # traverse layers backwards
        for layer in reversed(self.layers):
            gradient = layer.compute_gradient(gradient)
            
    def mini_batch_update(self, xs, ys, lr, l2, train_size):
        """Updates weights in the network.
        Args:
            xs (numpy.array): Training example inputs. Should have dimensions 
                (n, 1, in_size) where n is the number of training examples.
            ys (numpy.array): Training example outputs. Should have dimensions 
                (n, 1, layer_sizes[-1]).
            lr (float): Learning rate.
        """
        batch_size = ys.shape[0]

        for x, y in zip(xs, ys):
            y_pred = self.feedforward(x.reshape(1, -1))
            loss_grad = self.loss_diff(y, y_pred)
            self.backpropagate(loss_grad)
            
        for layer in reversed(self.layers):
            # update weights in layer
            layer.update_weights(lr=lr, train_size=train_size, 
                                 batch_size=batch_size, l2=l2)
            # reinitialize gradients
            layer.init_gradients()

    def score(self, x_test, y_test_labels):
        """Computes model accuracy.
        """
        y_pred = np.array([self.predict(x) for x in x_test])
        return np.mean(y_pred == y_test_labels)
        
    def compute_loss(self, xs, ys):
        """Computes model loss.
        """
        activations = np.array([self.feedforward(x) for x in xs]).reshape(xs.shape[0], -1)
        loss = np.array([self.loss_function(yt, yp) 
                        for yt, yp in zip(ys, activations)]).mean()   
        return loss

    def fit(self, x_train, y_train,  
            x_test=None, y_test=None,
            batch_size=32, epochs=20, lr=0.01, l2=0.0,
            compute_loss=True,
            compute_accuracy=False,
            verbose=0):
        """Fit the network using stochastic gradient descent.

        Fits the network to the training data. If test data are provided and `compute_performance` 
        set to True the loss and accuracy is evaluated on it after each training epoch.

        x_train (numpy.array): Training example inputs. Should have dimensions 
            (n, 1, in_size) where n is the number of training examples.
        y_train (numpy.array): Training example outputs. Should have dimensions 
            (n, 1, layer_sizes[-1]).
        x_test (numpy.array): Test example outputs. Should have dimensions 
            (n, 1, layer_sizes[-1]). Optional.
        y_test (numpy.array): Test example outputs. Should have dimensions 
            (n, 1, layer_sizes[-1]). Optional.
        batch_size (int): Number of training examples in each batch.
            Default is 32
        epochs (int): Number of times to pass through the training data. 
            Default is 20.
        lr (float): Learning rate. Default is 0.01
        l2 (float): L2 regularization strength.
        compute_performance (bool): The loss and accuracy on the training and test data 
            will be computed and stored in a dictionary if set to True. Default is False.
        verbose (int) The loss and accuracy on the training and test data 
            will be printed if set to 1. Default is 0

        Returns:
            history (dict(list)): The loss and accuracy on the training and test data 
                for each training epoch.
        """
        
        history = dict(
            train_loss=[],
            train_accuracy=[],
            test_loss=[],
            test_accuracy=[],
        )

        if compute_accuracy:
            y_test_labels = np.argmax(y_test, axis=1)
            y_train_labels = np.argmax(y_train, axis=1) 
                
        # number of rows in the data
        n_obs = y_train.shape[0]
        indices = np.arange(n_obs)
        # number of epochs
        for i in range(epochs):
            # shuffle the indices so that data is passed in 
            # random order to the mini-batch update
            np.random.shuffle(indices)
            # create the batches
            for j in range(0, n_obs, batch_size):
                batch_ix = indices[j: j + batch_size]
                
                # select data using the batch index
                # compute the gradients, and update the weights
                self.mini_batch_update(x_train[batch_ix, :], y_train[batch_ix, :], 
                                       lr=lr, l2=l2,
                                       train_size=n_obs)
            
            if compute_loss:
                # compute loss and accuracy on training (and test data if provided)
                history['train_loss'].append(self.compute_loss(x_train, y_train))   
                if verbose:
                    print("Epoch {}: Training loss: {:.4f}".format(i, history['train_loss'][-1]))
            
            if compute_accuracy:
                history['train_accuracy'].append(self.score(x_train, y_train_labels))
                if verbose:
                    print("\t Training accuracy: {:.4f}".format(history['train_accuracy'][-1]))

            if np.shape(x_test) and np.shape(y_test):                         
                if compute_loss:
                    history['test_loss'].append(self.compute_loss(x_test, y_test)) 
                    if verbose:
                        print("\t Test loss: {:.4f}".format(history['test_loss'][-1]))

                if compute_accuracy:
                    history['test_accuracy'].append(self.score(x_test, y_test_labels))
                    if verbose:
                        print("\t Test accuracy: {:.4f}".format(history['test_accuracy'][-1]))

        return history

    def __repr__(self):
        return str(self.layers)