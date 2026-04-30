from sklearn.datasets import make_blobs
import numpy as np
from model import *
import matplotlib.pyplot as plt
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
train_acc = np.mean(y_train_pred ==  y_train)
test_acc = np.mean(y_test_pred == y_test )


print(f"First loss: {loss_history[0]:.4f}")
print(f"Last loss: {loss_history[-1]:.4f}")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# We solve for x2 because x1 is the horizontal axis and x2 is the vertical axis.
# To draw a line, we generate x1 values and calculate the corresponding x2 values.

# np.linspace(start, stop, num=50, endpoint=True, retstep=False
x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
decision_boundary = -(w[0] * x1_vals + b) / w[1]
margin_positive = -(w[0] * x1_vals + b - 1) / w[1]
margin_negative = -(w[0] * x1_vals + b + 1) / w[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", label="train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker="x", label="test")

plt.plot(x1_vals, decision_boundary, "k-", label="decision boundary")
plt.plot(x1_vals, margin_positive, "k--", label="margin +1")
plt.plot(x1_vals, margin_negative, "k--", label="margin -1")

plt.legend()
plt.grid(True)
plt.savefig("svm.png")
plt.show()