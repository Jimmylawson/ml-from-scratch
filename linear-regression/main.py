# linear regression practice
from sklearn.datasets import fetch_california_housing
import numpy as np
from model import (predict, compute_cost, compute_gradient, fit_gradient_descent,mse,
                   error, normal_equation)


# dummy data
# X = housing.data
# y = housing.target

# print the shape of the data
# print(X.shape, y.shape)

#print the first 5 row of the data
# print(X[:5])
# print(y[:5])

# X = np.array(
#     [[1,2,3],
#     [3,4,5]]
# )
# y = np.array([1.0,3.0])
# theta = np.array([0.1, 0.2, 0.3])
# pred = predict(X, theta)
# print(pred)
# print(f" m, n = {X.shape}: X.shape")
# print(f" theta shape = {theta.shape}")
# print(f" pred shape = {pred.shape}")
# print(f"The cost function is {compute_cost(X, y, theta)}")
# print(f"The gradient is {compute_gradient(X, y, theta)}")
# alpha = 0.01
# num_iters = 2000
#
# final_theta, cost_history = fit_gradient_descent(X, y, theta, alpha, num_iters)
# print(f"Final theta: {final_theta}")
# print(f"Cost history (first 10): {cost_history[:10]}")
# print(f"Final cost: {cost_history[-1]}")

housing = fetch_california_housing()

X, y = housing.data, housing.target

rg = np.random.default_rng(42)
idx = rg.permutation(X.shape[0])

X = X[idx]  # House #0 features move to position 12345
y = y[idx]  # House #0 price also moves to position 12345

#split the data to train and test
m = X.shape[0]
m_train = int(0.8 * m)
X_train = X[:m_train]
X_test = X[m_train:]
y_train = y[:m_train]
y_test = y[m_train:]


#Standard features to make the gradient descent work properly
mu = X_train.mean(axis=0) #axis means compute down the rows for each column

sigma = X_train.std(axis=0) # sigma stores the std for each  feature
sigma[sigma == 0] = 1.0 # help us not to divide by zero
#Before Standardization
# print(f"Original means: {mu}", mu)
# print(f"Original stds: {sigma}", sigma)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

#AFter Standardization
# print(f"Standardized means: {X_train.mean(axis=0)}")
# print(f"Standardized stds: {X_train.std(axis=0)}")


#add bias column of ones to the training and test data meaning it will create theta  which is 1 by default
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
# print(f"shape of X_train is {X_train.shape}")
# print(f"shape of X_test is {X_test.shape}")

theta = np.zeros(X_train.shape[1])
#gradient descent will update the theta with the optimal values
#cos our goal is find theta values that minimize the prediction error(cost function)
#0.01: relatively small step size
alpha = 0.01 # learning rate. How big of the steps to take when updating theta.
num_iters=2000 # number of iterations which controls how many times gradient descent updates the theta

theta,cost_history = fit_gradient_descent(
    X_train,
    y_train,
    theta,
    alpha,
    num_iters
);  # Add semicolon to suppress output
# print(f"First cost: {cost_history[0]}")      # Should be high
# print(f"Last cost: {cost_history[-1]}")      # Should be low
# print(f"Cost improvement: {cost_history[0] - cost_history[-1]}")
#

#Evaluation of unseen data
y_train_pred = predict(X_train,theta)
y_test_pred = predict(X_test,theta)

#Why this step matters:
#Gradient descent tells you optimization worked.
#Train vs test MSE tells you generalization quality.
train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)


print(f"MSE on train: {train_mse:.6f}")
print(f"MSE on test: {test_mse:.6f}")

#next is the RMSE
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(f"RMSE on train: {train_rmse:.6f}")
print(f"RMSE on test: {test_rmse:.6f}")

#next is R2 is see how it works
train_sse = np.sum(error(y_train, y_train_pred) ** 2)
train_sst = np.sum(error(y_train, np.mean(y_train)) ** 2)
train_r2 = 1 - (train_sse / train_sst)

test_sse = np.sum(error(y_test, y_test_pred) ** 2)
test_sst = np.sum(error(y_test, np.mean(y_test)) ** 2)
test_r2 = 1 - (test_sse / test_sst)
print(f"R2 on train: {train_r2:.6f}")
print(f"R2 on test: {test_r2:.6f}")

#normal equation
print("------ NORMAL EQUATION -----")
theta_ne = normal_equation(X_train, y_train)
y_train_ne = predict(X_train, theta_ne)
y_test_ne = predict(X_test, theta_ne)

train_mse_ne = mse(y_train, y_train_ne)
test_mse_ne = mse(y_test, y_test_ne)
print(f"MSE on train (normal equation): {train_mse_ne:.6f}")
print(f"MSE on test (normal equation): {test_mse_ne:.6f}")

train_rmse_ne = np.sqrt(train_mse_ne)
test_rmse_ne = np.sqrt(test_mse_ne)
print(f"RMSE on train (normal equation): {train_rmse_ne:.6f}")
print(f"RMSE on test (normal equation): {test_rmse_ne:.6f}")

train_sse_ne = np.sum((y_train - y_train_ne) ** 2)
train_sst_ne = np.sum((y_train - y_train.mean()) ** 2)
train_r2_ne = 1 - (train_sse_ne / train_sst_ne)

test_sse_ne = np.sum((y_test - y_test_ne) ** 2)
test_sst_ne = np.sum((y_test - y_test.mean()) ** 2)
test_r2_ne = 1 - (test_sse_ne / test_sst_ne)

print(f"R2 on train (normal equation): {train_r2_ne:.6f}")
print(f"R2 on test (normal equation): {test_r2_ne:.6f}")

