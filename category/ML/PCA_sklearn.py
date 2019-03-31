import pandas as pd
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import tensorflow.examples.tutorials.mnist as mn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# # # ------------ PCA on iris dataset ------------ # # #
# # # Loading the iris dataset from the link
# df = pd.read_csv(
#     filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
#     header=None,
#     sep=',')
#
# df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
# df.dropna(how="all", inplace=True)  # drops the empty line at file-end
# df.tail()
#
# # split data table into data X and class labels y
# X = df.iloc[:, 0:4].values  # ndarray with shape=(150, 4)
# y = df.iloc[:, 4].values  # ndarray with shape=(150,)
#
# # # Colors and Markers
# colors = {'Iris-setosa': '#0D76BF',
#           'Iris-versicolor': '#00cc96',
#           'Iris-virginica': '#EF553B'}
#
# markers = {'Iris-setosa': 'o',
#            'Iris-versicolor': 'v',
#            'Iris-virginica': 's'}
#
# # # Standardizing
# X_std = StandardScaler().fit_transform(X)
#
# # # PCA using sklearn
# sklearn_pca = sklearnPCA(n_components=2)
# # sklearn_pca = sklearnPCA(n_components=3)
# Y_sklearn = sklearn_pca.fit_transform(X_std)
#
# # # Plot the result in 2D.
# data = []
# for name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
#     trace = dict(
#         label=name,
#         data=Y_sklearn[y == name, :]
#     )
#     data.append(trace)
#
# for i in range(len(data)):
#     data_x = data[i]['data'][:, 0].ravel()
#     data_y = data[i]['data'][:, 1].ravel()
#     plt.scatter(data_x, data_y, c=colors[data[i]['label']], label=data[i]['label'], marker=markers[data[i]['label']])
#
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.legend(loc='best')
# plt.show()
#
# # # Plot the result in 3D.
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # data = []
# # for name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
# #     trace = dict(
# #         label=name,
# #         data=Y_sklearn[y == name, :]
# #     )
# #     data.append(trace)
# #
# # for i in range(len(data)):
# #     label = data[i]['label']
# #     data_x = data[i]['data'][:, 0].ravel()
# #     data_y = data[i]['data'][:, 1].ravel()
# #     data_z = data[i]['data'][:, 2].ravel()
# #     ax.scatter(data_x, data_y, data_z, c=colors[label], label=label, marker=markers[label])
# #
# # ax.set_xlabel('X Label')
# # ax.set_ylabel('Y Label')
# # ax.set_zlabel('Z Label')
# # plt.legend(loc='best')
# # plt.show()

# # # ------------ PCA on iris dataset ------------ # # #


# # # ------------ PCA on MNIST dataset ------------ # # #
# # Loading the MNIST dataset.
mnist = mn.input_data.read_data_sets("./dataset/mnist/", one_hot=False)
X = mnist.train.images
y = mnist.train.labels
# # Colors and Markers
colors = {0: '#FF5733',
          1: '#FCFF33',
          2: '#4CFC1C',
          3: '#09FEE2',
          4: '#D912D9',
          5: '#000000',
          6: '#125FD9',
          7: '#2F9A5F',
          8: '#CBFF33',
          9: '#FFA833'}

markers = {0: '.',
           1: 'v',
           2: '^',
           3: '<',
           4: '>',
           5: 's',
           6: 'p',
           7: 'P',
           8: '+',
           9: 'x'}

# # Standardizing
X_std = StandardScaler().fit_transform(X)

# # PCA using sklearn
sklearn_pca = sklearnPCA(n_components=2)
# sklearn_pca = sklearnPCA(n_components=3)
Y_sklearn = sklearn_pca.fit_transform(X_std)

# # Plot the result in 2D.
data = []
for lab in range(10):
    trace = dict(
        label=lab,
        data=Y_sklearn[y == lab, :]
    )
    data.append(trace)

for i in range(len(data)):
    data_x = data[i]['data'][:, 0].ravel()
    data_y = data[i]['data'][:, 1].ravel()
    plt.scatter(data_x, data_y, c=colors[data[i]['label']], label=data[i]['label'], marker=markers[data[i]['label']])

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='best')
plt.show()

# # Plot the result in 3D.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# data = []
# for lab in range(10):
#     a = Y_sklearn[y == lab, :]
#     trace = dict(
#         label=lab,
#         data=Y_sklearn[y == lab, :]
#     )
#     data.append(trace)
#
# for i in range(len(data)):
#     label = data[i]['label']
#     data_x = data[i]['data'][:, 0].ravel()
#     data_y = data[i]['data'][:, 1].ravel()
#     data_z = data[i]['data'][:, 2].ravel()
#     ax.scatter(data_x, data_y, data_z, c=colors[label], label=label, marker=markers[label])
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.legend(loc='best')
# plt.show()

# # # ------------ PCA on MNIST dataset ------------ # # #
