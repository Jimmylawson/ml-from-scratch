from sklearn.datasets import load_iris
import numpy as np
from model import *

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

#Print the shape of the data
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

#print the first 5 rows of the data
# print(X[:5])
# print(y[:5])

rg = np.random.default_rng(42)
idx = rg.permutation(X.shape[0])
X = X[idx]
y = y[idx]

#number of features
d = X.shape[1]
print(f"Number of features: {d}")

#80% for training set and 20 for testing
m = X.shape[0]
m_train = int(0.8 * m)
X_train = X[:m_train]
X_test = X[m_train:]
y_train = y[:m_train]
y_test = y[m_train:]

priors, classes = class_prior(y_train)
mus = mean_vector(X_train, y_train, classes)
sigma = covariance_matrix(X_train, y_train, classes, mus)

y_pred = np.array([predict_one(x, priors,classes, mus,sigma) for x in X_test])
acc = np.mean(y_pred == y_test)
print(f"Accuracy: {acc:.4f}")