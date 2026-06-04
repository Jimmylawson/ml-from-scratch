import numpy as np

from keras.datasets import mnist
from net_model import (one_hot, mini_batch_gradient,prediction,accuracy,forward_prop)

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

#Transform the matrix
X_train = X_train.T # our neural net expect each column = one image
X_test = X_test.T
# print(X_train.shape) #(784, 60000)
# print(X_test.shape) #(784, 10000)
# print(X_train[:5, :3])
#preprocess y
Y_train = one_hot(y_train)
Y_test  = one_hot(y_test)
# print(y_train[0])
# print(Y_train[:, 0])
# print(Y_train.shape)

#train
W1,b1,W2,b2 = mini_batch_gradient(X_train,Y_train, B=64, alpha=0.1,num_iteration=10)


_, _, _, A2_train = forward_prop(W1,b1,W2,b2,X_train)
train_predictions = prediction(A2_train)
train_accuracy = accuracy(train_predictions ,y_train)

_, _, _, A2_test = forward_prop(W1,b1,W2,b2,X_test)
test_prediction = prediction(A2_test)
test_accuracy = accuracy(test_prediction, y_test)

print("Train accuracy: ", train_accuracy)
print("Test accuracy:", test_accuracy)

print("First 10 predictions: ", test_prediction[:10])
print("First 10 labels: ", y_test[:10])