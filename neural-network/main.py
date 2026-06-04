import numpy as np

from keras.datasets import mnist
from net_model import (one_hot)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ---- CHECKING THE MNIST LOAD DATA  -------
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_train[0].shape)
# print(y_train[0])

# ---- RESHUFFLING DATA -----
#To help neural net train better because the input numbers are smaller and more stable
X_train = X_train.reshape(60000,784) /255.0
X_test = X_test.reshape(10000,784) /255.0

#Transform the mattrix
X_train = X_train.T
X_test = X_test.T
# print(X_train.shape) #(784, 60000)
# print(X_test.shape) #(784, 10000)
# print(X_train[:5, :3])

Y_train = one_hot(y_train)
Y_test  = one_hot(y_test)

print(y_train[0])
print(Y_train[:, 0])


