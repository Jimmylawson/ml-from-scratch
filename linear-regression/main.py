# linear regression practice
from sklearn.datasets import fetch_california_housing
import numpy as np
from model import (predict,fit_gradient_descent,mse, normal_equation,rmse,r2)
import matplotlib.pyplot as plt
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

theta = np.zeros(X_train.shape[1]) #initialize theta with zeros
#gradient descent will update the theta with the optimal values
#cos our goal is find theta values that minimize the prediction error(cost function)
#0.01: relatively small step size
alpha = 0.01 # learning rate. How big of the steps to take when updating theta.
num_iter=2000 # number of iterations which controls how many times gradient descent updates the theta

theta,cost_history = fit_gradient_descent(
    X_train,
    y_train,
    theta,
    alpha,
    num_iter
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
train_rmse = rmse(train_mse)
test_rmse = rmse(test_mse)

print(f"RMSE on train: {train_rmse:.6f}")
print(f"RMSE on test: {test_rmse:.6f}")

#next is R2 is see how it works

train_r2_gd = r2(y_train,y_train_pred)
test_r2_gd = r2(y_test,y_test_pred)
print(f"R2 on train: {train_r2_gd:.6f}")
print(f"R2 on test: {test_r2_gd:.6f}")

#normal equation
print("------ NORMAL EQUATION -----")
theta_ne = normal_equation(X_train, y_train)
#doing a comparison with gradient descent
iter_list = [100,500,2000,10000]
train_mse_list = []
test_mse_list = []
theta_gap_list = []

print("iters | train_mse | test_mse | theta_gap_to_NE")
for num_iters in iter_list:
    theta0 = np.zeros(X_train.shape[1])
    theta_gd, _ = fit_gradient_descent(X_train,y_train,theta0,alpha,num_iters)

    y_train_pred = predict(X_train,theta_gd)
    y_test_pred = predict(X_test,theta_gd)

    train_mse = mse(y_train,y_train_pred)
    test_mse = mse(y_test,y_test_pred)
    theta_gap = np.linalg.norm(theta_gd - theta_ne)
    print(f"{num_iters:5d} | {train_mse:.6f} | {test_mse:.6f} | {theta_gap:.6f}")
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    theta_gap_list.append(np.linalg.norm(theta_gd - theta_ne))
plt.plot(iter_list, train_mse_list, marker="o",label="train")
plt.plot(iter_list, test_mse_list,marker="o", label="test")
plt.xlabel("num_iters") # Label X-axis
plt.ylabel("MSE") # Label Y-axis
plt.title("GD Convergence (MSE vs Iteration)") #Adds plot title
plt.legend() #show the labels("train", "test")
plt.grid(True) # Adds grid lines
# plt.xscale("log") # Log scale for x-axis (commented out for linear scale)
plt.savefig("gd_convergence.png") #Save the plot
plt.show() #Display the plot



# y_train_ne = predict(X_train, theta_ne)
# y_test_ne = predict(X_test, theta_ne)
#
# train_mse_ne = mse(y_train, y_train_ne)
# test_mse_ne = mse(y_test, y_test_ne)
# print(f"MSE on train (normal equation): {train_mse_ne:.6f}")
# print(f"MSE on test (normal equation): {test_mse_ne:.6f}")
#
# train_rmse_ne = rmse(train_mse_ne)
# test_rmse_ne = rmse(test_mse_ne)
# print(f"RMSE on train (normal equation): {train_rmse_ne:.6f}")
# print(f"RMSE on test (normal equation): {test_rmse_ne:.6f}")
#
# train_r2_ne = r2(y_train, y_train_ne)
# test_r2_ne = r2(y_test,y_test_ne)
#
# print(f"R2 on train (normal equation): {train_r2_ne:.6f}")
# print(f"R2 on test (normal equation): {test_r2_ne:.6f}")

#showing the convergence of the gradient descent and N
#