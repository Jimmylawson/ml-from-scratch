import numpy as np




def predict(X,theta):
    return X @ theta

def sigmoid(X, theta):
    z = predict(X,theta)
    return 1 / (1 + np.exp(-z))

def logistic_cost_function(X,theta,y):
    y_hat = sigmoid(X,theta)
    m = len(y)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon) #help to print the log from infinite to a small number
    cost =  - (1/ m) * np.sum(
        y * np.log(y_hat) + (1 - y) * np.log( 1 - y_hat))
    return cost

def logistic_gradient(X,y, theta):
    m = len(y)
    error = sigmoid(X, theta) - y
    gradient = (1/m )  * X.T @ error

    return gradient

def fit_gradient_descent(X, y, theta, num_iters, alpha):
    cost_history = []

    for i in range(num_iters):
        grad = logistic_gradient(X, y, theta)
        theta = theta - alpha * grad

        cost_function = logistic_cost_function(X, theta, y)
        cost_history.append(cost_function)

    return theta, cost_history

def pred_prob(X, theta):
    return sigmoid(X, theta)

#predict_class convert probably to class label( 0 or 1 )
def predict_class(X, theta,threshold=0.5):
    prob = pred_prob(X, theta)
    return (prob >= threshold).astype(int)

#accuracy is super import to evaluate your model
def accuracy(X, y, theta):
    preds = predict_class(X, theta)
    return np.mean(preds == y)

# --- Classification Metrics ---
def precision(TP, FP):
    return TP / (TP + FP) if(TP + FP) > 0 else 0.0

def recall(TP, FN):
    return TP / (TP + FN) if(TP + FN) > 0 else 0.0

def f1_score(TP, FP, FN):
    p = precision(TP, FP)
    r = recall(TP, FN)

    return 2 * (p * r) / (p + r)