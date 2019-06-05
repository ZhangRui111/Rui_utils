import tensorflow.examples.tutorials.mnist as mn

# # Loading the MNIST dataset.
mnist = mn.input_data.read_data_sets("./dataset/mnist/", one_hot=False)
X = mnist.train.images
y = mnist.train.labels

# # # # # 1. sklearn shuffle. # # # # #
from sklearn.utils import shuffle

X, y = shuffle(X, y)  # (55000, 784), (55000,)

print(X.shape, y.shape)

# # # # # 2. numpy shuffle. # # # # #
import numpy as np

y = y[:, np.newaxis]
dataset = np.concatenate((X, y), axis=1)
print(dataset.shape)  # (55000, 785)

np.random.shuffle(dataset)  # np.random.shuffle() doesn't have return value.

X = dataset[:, :-1]
y = dataset[:, -1]
print(X.shape, y.shape)  # (55000, 784), (55000,)
