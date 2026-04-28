from sklearn.datasets import make_blobs
import numpy as np
from model import *

X, y = make_blobs(
    n_samples=100,
    n_features=2,
    centers=2,
    cluster_std=1.0,
    random_state=42
)

rng = np.random.default_rng()
indx = rng.permutation(len(X))
#np.where(condition, value_if_true, value_if_false)
y_svm= np.where(y == 0, -1 ,1 )
X = X[indx]
y_svm = y_svm[indx]


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_svm[:split], y_svm[split:]

print(f"X shape {X_train.shape}")
print(f"y shape {y_train.shape}")


print(y[:5])
print(y_svm[:5])

