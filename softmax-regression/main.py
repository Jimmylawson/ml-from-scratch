from sklearn.datasets import load_digits
import numpy as np

data = load_digits()

X, y  = data.data, data.target

# print(X.shape, y.shape)

#shuffle the data
rg = np.random.default_rng(42)
idx = rg.permutation(len(X))
X, Y = X[idx], y[idx]

m = len(X)
m_train = int(0.8 * m)
X_train,X_test = X[:m_train],X[m_train:]
y_train,y_test = Y[:m_train],Y[m_train:]


#Standard deviation and mean to make gradient descent work better
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma== 0] = 1 # help us not to divide by zero
print(f"Original means: {mu}", mu)
print(f"Original stds: {sigma}", sigma)
X_train = (X_train  - mu) / sigma
X_test = (X_test - mu) / sigma


#add bias column to the training and the test
X_train = np.hstack([np.ones((X_train.shape[0], 1)),X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)),X_test])