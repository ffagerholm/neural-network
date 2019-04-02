#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `neural_network` package."""


import unittest
import numpy as np

from neural_network import neural_network

class TestLayer(unittest.TestCase):
    """Tests for `Layer` class."""

    def setUp(self):
        self.in_shape = np.random.randint(1, 20)
        self.out_shape = np.random.randint(1, 20)
        self.layer = neural_network.Layer(self.in_shape, self.out_shape)
        self.input = np.random.uniform(size=(1, self.in_shape))

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_layer_shapes(self):
        """Test that the weight and bias are of the correct shapes."""
        self.assertEqual(self.layer.weights.shape, (self.in_shape, self.out_shape))
        self.assertEqual(self.layer.bias.shape, (1, self.out_shape))

    def test_activation_shape(self):
        """Test that the activation has the correct shape."""
        activation = self.layer.compute_activation(self.input)
        self.assertEqual(activation.shape, (1, self.out_shape))

    def test_activation_value(self):
        """Test that the activation has the correct value."""
        activation = self.layer.compute_activation(self.input)
        # activations should be between 0 and 1 because of the sigmoid function
        self.assertTrue(np.all((activation >= 0.0) & (activation <= 1.0)))


class TestNeuralNetwork(unittest.TestCase):
    """Tests for `NeuralNetwork` class."""

    def setUp(self):
        self.num_layers = np.random.randint(3, 11)
        self.layer_shapes = np.random.randint(1, 20, size=self.num_layers + 1)
        self.neural_network = neural_network.NeuralNetwork(layer_sizes)
        
    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_number_of_layers(self):
        """Test that network has correct number of layers."""
        self.assertEqual(len(self.neural_network.layers), self.num_layers) 


if __name__ == '__main__':
    unittest.main()