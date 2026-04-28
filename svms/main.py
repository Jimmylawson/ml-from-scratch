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

rng = np.random.default_rng(42)
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

w, b, loss_history = fit_svm(X_train, y_train, alpha=0.01, C=1.0, num_iters=1000)
y_train_pred= predict_class(X_train, w, b)
y_test_pred  = predict_class(X_test, w, b)

# How many predicted labels matched the real labels?
train_acc = np.mean(y_train == y_train)
test_acc = np.mean(y_test == y_test)


print(f"First loss: {loss_history[0]:.4f}")
print(f"Last loss: {loss_history[-1]:.4f}")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")