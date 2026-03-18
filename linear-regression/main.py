# linear regression practice
from sklearn.datasets import fetch_california_housing
import numpy as np
from model import predict, compute_cost

housing = fetch_california_housing()
#
# X = housing.data
# y = housing.target

# print the shape of the data
# print(X.shape, y.shape)

#print the first 5 row of the data
# print(X[:5])
# print(y[:5])

X = np.array(
    [[1,2,3],
    [3,4,5]]
)
y = np.array([1.0,3.0])
theta = np.array([0.1, 0.2, 0.3])
pred = predict(X, theta)
print(pred)

print(f" m, n = {X.shape}: X.shape")
print(f" theta shape = {theta.shape}")
print(f" pred shape = {pred.shape}")
print(f"The cost function is {compute_cost(X, y, theta)}")

