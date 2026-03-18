import numpy as np

def predict(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    err  = predict(X, theta) - y
    print(err)
    m = len(y)
    cost_function =  (1 /(2 * m)) * np.sum(err ** 2)

    return cost_function


def compute_gradient(X, y, theta):
