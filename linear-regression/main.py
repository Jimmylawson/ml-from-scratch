# linear regression practice
from sklearn.datasets import fetch_california_housing
import numpy as np
from model import predict, compute_cost, compute_gradient, fit_gradient_descent


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

# Create a random number generator with a fixed seed so results are reproducible
rng = np.random.default_rng(42)

# Build a shuffled list of row indices: [0, 1, ..., m-1] in random order
idx = rng.permutation(X.shape[0])

# Reorder X using the shuffled indices
X = X[idx]

# Reorder y with the exact same indices to keep (x_i, y_i) pairs aligned
y = y[idx]


#Number of total examples
m = X.shape[0]
#80% for training
m_train = int(0.8 * m)

#First 80%  -> train, remaining 20% -> test
# split the data into training and testing sets
X_train = X[:m_train]
y_train = y[:m_train]
X_test = X[m_train:]
y_test = y[m_train:]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


#Compute feature-wise mean/std from TRAIN only
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

# Safety: avoid divide-by-zero if any feature has zero variance
sigma[sigma == 0] = 1.0

#Standardize train and test using TRAIN stats
X_train = (X_train -  mu) /sigma
X_test = (X_test - mu)  / sigma



# Add bias column (x0 = 1) as first column
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print("X_train with bias:", X_train.shape)
print("X_test with bias :", X_test.shape)

theta0 = np.zeros(X_train.shape[1])
theta, cost_history = fit_gradient_descent(
    X_train, y_train, theta0, alpha=0.01, num_iters=2000
)

print("first cost:", cost_history[0])
print("last cost :", cost_history[-1])
