import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pickle
import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from plotting import plot_confusion_matrix, plot_training_history

with gzip.open('data/mnist.pkl.gz', 'rb') as infile:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(infile, encoding="latin1")


x_train = x_train.reshape(-1, 1, 28*28).astype(np.float32)
y_train = y_train.reshape(-1, 1)

x_val = x_val.reshape(-1, 1, 28*28).astype(np.float32)
y_val = y_val.reshape(-1, 1)

x_test = x_test.reshape(-1, 1, 28*28).astype(np.float32)
y_test = y_test.reshape(-1, 1)


encoder = OneHotEncoder(categories='auto', sparse=False)
y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

assert (np.argmax(y_train_encoded, axis=1) == y_train.ravel()).all()
assert (np.argmax(y_val_encoded, axis=1) == y_val.ravel()).all()

network = NeuralNetwork([784, 16, 10], activation_functions=['sigmoid', 'softmax'])

history = network.fit(x_train, y_train_encoded, 
                      x_val, y_val_encoded,
                      batch_size=32, epochs=20, lr=1.5, l2=1.0,
                      verbose=1, 
                      compute_loss=True,
                      compute_accuracy=True)


plot_training_history(history['train_loss'], history['test_loss'], 
                      history['train_accuracy'], history['test_accuracy'])

# evaluate model on test data
y_pred = np.array([network.predict(x) for x in x_test])
plot_confusion_matrix(y_test, y_pred, classes=np.arange(10), normalize=True)

plt.show()