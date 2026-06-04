import numpy as np

# 784 = pixels in one image
# 60000 = number of training images
# 10 = possible digit labels/classes

def one_hot(y, num_classes = 10):
    one_hot_y = np.zeros((num_classes, y.size))
    one_hot_y[y,np.arange(y.size)] = 1
    return one_hot_y


def init_params():
#0.01 makes the weight small
    hidden_size = 128
    W1 = np.random.randn(hidden_size, 60000) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(10,hidden_size) * 0.01
    b2 = np.zeros((10,1))

    return  W1, b1, W2, b2

def RELU(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = RELU(Z1)
    Z2 = W2.dot(A1)  + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def relu_derivative(z):
    return z > 0

def backprop(Z1,A1,Z2,A2,W2,X,Y):
    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2  @ A1.T
    db2 =  (1 / m ) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @dZ2 *relu_derivative(Z1)
    dW1 = (1 / m ) * dZ1 * X.T
    db1 = (1/m)  * np.sum(dZ1, axis = 1, keepdims=True)

