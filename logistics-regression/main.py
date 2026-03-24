import numpy as np
from sklearn.datasets import load_breast_cancer

cancer  = load_breast_cancer()

X, Y = cancer.data, cancer.target

# print(X.shape, Y.shape)

#reshuffling the data
rg = np.random.default_rng(42)
idx = rg.permutation(X.shape[0])
X = X[idx]
Y = Y[idx]

#split data into train and test data
m = X.shape[0]
m_train = int(0.8 * m)
X_train = X[:m_train]
X_test = X[m_train:]
Y_train = Y[:m_train]
Y_test = Y[m_train:]

#standard deviation and mean to make gradient descent work better
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1 #helps us not to divide by zero
print(f"Original means: {mu}", mu)
print(f"Original stds: {sigma}", sigma)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
#After Standardization
print(f"Standardized means: {X_train.mean(axis=0)}")
print(f"Standardized stds: {X_train.std(axis=0)}")

# add bias column of one to the training and the test set to make gradient descent work
X_train = np.hstack([np.ones(X_train.shape[0], 1),X_train])
X_test = np.hstack([np.ones(X_test.shape[0], 1), X_test])