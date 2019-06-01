# # # # # Loading the MNIST dataset. # # # # #
import tensorflow.examples.tutorials.mnist as mn

mnist = mn.input_data.read_data_sets("./dataset/mnist/", one_hot=False)
X_train = mnist.train.images
y_train = mnist.train.labels
# default validation size is 5000, which can be changed in
# line 237 of tensorflow.examples.tutorials.mnist.py
X_valid = mnist.validation.images
y_valid = mnist.validation.labels
X_test = mnist.test.images
y_test = mnist.test.labels

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# # # # # Loading the CIFAR10 dataset. # # # # #
from category.ML import extract_cifar10

cifar10_data_set = extract_cifar10.Cifar10DataSet('./dataset/cifar10/', one_hot=False)
train_images, train_labels = cifar10_data_set.train_data()
test_images, test_labels = cifar10_data_set.test_data()

# get batch data.
for i in range(60000):
    batch_xs, batch_ys = cifar10_data_set.next_train_batch(50)
    print(batch_xs.shape, batch_ys.shape)

# # # # # Loading the iris dataset. # # # # #
# 1. Loading the iris dataset from the link.
import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)  # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y
X = df.iloc[:, 0:4].values  # ndarray with shape=(150, 4)
y = df.iloc[:, 4].values  # ndarray with shape=(150,)

print(X.shape, y.shape)

# 2. Loading the iris dataset from local file.
df = pd.read_csv('./dataset/iris/iris.csv', header=None, sep=',')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)  # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y
X = df.iloc[1:, 0:4].values  # ndarray with shape=(150, 4)
y = df.iloc[1:, 4].values  # ndarray with shape=(150,)

print(X.shape, y.shape)

# Or in numpy way.
from numpy import genfromtxt

raw = genfromtxt('./dataset/iris/iris.csv', delimiter=',', dtype=str)

# split data table into data X and class labels y
X = raw[1:, 0:4]  # ndarray with shape=(150, 4)
y = raw[1:, 4]  # ndarray with shape=(150,)

print(X.shape, y.shape)
