import numpy as np

def predict(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    err = error(predict(X, theta), y)
    print(err)
    m = len(y)
    cost_function =  (1 /(2 * m)) * np.sum(err ** 2)

    return cost_function

def error(predict_values, y):
    return predict_values - y

def compute_gradient(X, y, theta):
    m = len(y)
    err = error(predict(X, theta), y)
    gradient = (1 / m ) * X.T @ err
    return gradient

def fit_gradient_descent(X,y, theta, alpha, num_iters):
    cost_history = []
    for _ in range(num_iters):
        grad = compute_gradient(X, y, theta)
        update_theta = theta - alpha * grad
        cost = compute_cost(X, y, update_theta)
        cost_history.append(cost)
        theta = update_theta
    return theta, cost_history
