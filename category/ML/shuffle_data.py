import tensorflow.examples.tutorials.mnist as mn

from sklearn.utils import shuffle

# # Loading the MNIST dataset.
mnist = mn.input_data.read_data_sets("./dataset/mnist/", one_hot=False)
X = mnist.train.images
y = mnist.train.labels

X, y = shuffle(X, y)

print(X.shape, y.shape)
