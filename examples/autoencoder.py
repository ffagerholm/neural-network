"""
Autoencoder for MNIST images.
"""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from neural_network import NeuralNetwork


with gzip.open('data/mnist.pkl.gz', 'rb') as infile:
    (x_train, y_train), _, (x_test, y_test) = pickle.load(infile, encoding="latin1")

x_train = x_train.reshape(-1, 1, 28*28).astype(np.float32)
y_train = y_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1, 28*28).astype(np.float32)
y_test = y_test.reshape(-1, 1)

autoencoder = NeuralNetwork(layer_sizes=[784, 64, 32, 64, 784])
autoencoder.fit(x_train, x_train, epochs=30, lr=0.8, 
                compute_loss=True, verbose=1)

n_test = x_test.shape[0] 

random_index = np.random.choice(np.arange(n_test), 16) 

column1 = gridspec.GridSpec(4, 4)
column1.update(left=0.05, right=0.45, wspace=0.1)

column2 = gridspec.GridSpec(4, 4)
column2.update(left=0.55, right=0.95, wspace=0.1)

plt.figure(figsize=(12, 12))
plt.suptitle("Real vs decoded images")

for ix, gs1, gs2 in zip(random_index, column1, column2):
    x_true = x_test[ix]
    x_pred = autoencoder.feedforward(x_true)
    
    plt.subplot(gs1)
    plt.imshow(x_true.reshape(28, 28), cmap=plt.cm.gray_r)
    plt.tick_params(axis='both', 
                    which='both',
                    bottom=False, 
                    left=False,
                    labelbottom=False, 
                    labelleft=False,)

    plt.subplot(gs2)
    plt.imshow(x_pred.reshape(28, 28), cmap=plt.cm.gray_r)
    plt.tick_params(axis='both', 
                    which='both',
                    bottom=False, 
                    left=False,
                    labelbottom=False, 
                    labelleft=False,)

plt.show()
