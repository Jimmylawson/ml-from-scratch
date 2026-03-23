import numpy as np

def predict(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    err = error(predict(X, theta), y)
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

def fit_gradient_descent(X, y, theta, alpha, num_iters, log_every=None):
    """
    Batch Gradient Descent to find optimal theta values
    
    Parameters:
    X: Feature matrix with bias column (m_samples, n_features)
    y: Target values (m_samples,)
    theta: Initial parameters (n_features,)
    alpha: Learning rate (step size for updates)
    num_iters: Number of iterations to run
    
    Returns:
    theta: Final learned parameters
    cost_history: Cost value at each iteration (for convergence checking)
    """
    cost_history = []
    for i in range(num_iters):
        # Compute gradient using ALL training samples (batch gradient descent)
        grad = compute_gradient(X, y, theta)
        
        # Update theta: take step in opposite direction of gradient
        update_theta = theta - alpha * grad
        
        # Compute cost with new theta to track convergence
        cost_function = compute_cost(X, y, update_theta)
        cost_history.append(cost_function)
        if log_every is not None and (i % log_every == 0 or i == num_iters - 1):
            print(f"iter {i:4d} | cost {cost_function:.6f}")
        
        # Set theta to updated values for next iteration
        theta = update_theta
    return theta, cost_history

#the derivate approach to find the optimal theta value in one step
#using the normal equation
def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

#methods for evaluation the model to check its performance
def mse(y,y_pred):
    return np.mean((y- y_pred) ** 2)
def rmse(mse_value):
    return np.sqrt(mse_value)
def r2(y_train, y_pred):
    sse = np.sum((y_train - y_pred) ** 2)
    sst = np.sum((y_train - np.mean(y_train)) ** 2)
    return 1 - (sse / sst)