import numpy as np




def predict(X,theta):
    return X @ theta

def sigmoid(X, theta):
    z = predict(X,theta)
    return 1 / (1 + np.exp(-z))

def logistic_cost_function(X,theta,y):
    y_hat = sigmoid(X,theta)
    m = len(y)
    cost =  - (1/ m) *  (np.sum(y * np.log(y_hat)) + (1 - y) * np.log( 1 - y_hat))

    return cost

def gradient_descent(X,y, theta):
    m = len(y)
    error = sigmoid(X, theta) - y
    gradient = (1/m )  * X.T @ error

    return gradient

def fit_gradient_descent(X, y, theta, num_iters, alpha):
    cost_history = []

    for i in range(num_iters):
        grad = gradient_descent(X,y, theta)
        theta = theta - alpha * grad

        cost_function = logistic_cost_function(X, theta, y)
        cost_history.append(cost_function)

    return theta, cost_history

def pred_prob(X, theta):
    return sigmoid(X, theta)

#predict_class convert probably to class label( 0 or 1 )
def predict_class(X, theta):
    prob = pred_prob(X, theta)
    return (prob >= 0.5).astype(int)
