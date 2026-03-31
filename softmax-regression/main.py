from sklearn.datasets import load_digits
import numpy as np
from model import (fit_softmax_gd, predict_class, softmax,softmax_gradient,accuracy)

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
# print(f"Original means: {mu}", mu)
# print(f"Original stds: {sigma}", sigma)
X_train = (X_train  - mu) / sigma
X_test = (X_test - mu) / sigma


#add bias column to the training and the test
X_train = np.hstack([np.ones((X_train.shape[0], 1)),X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)),X_test])

alpha = 0.01
num_iters = 20000
log_every = 1000
num_classes = len(np.unique(y_train)) #count how many different classes your data contains
theta = np.zeros((X_train.shape[1],num_classes))

#fit gradient descent
theta, cost_history = fit_softmax_gd(X_train, y_train, theta, alpha, num_iters, log_every)

#evaluate
train_accuracy = accuracy(X_train, y_train, theta)
test_accuracy = accuracy(X_test, y_test, theta)

# print(f"Train accuracy: {train_accuracy:.4f}")
# print(f"Test accuracy: {test_accuracy:.4f}")

#THe output is a diagonal which show how our model performance
#
print("confusion matrix:")
y_pred = predict_class(X_test,theta)
cm = np.zeros((num_classes,num_classes),dtype=int)

for i in range(len(y_test)):
    actual = y_test[i]
    predicted = y_pred[i]
    cm[actual,predicted] += 1


print(cm)

print("Per-class accuracy:")
for c in range(num_classes):
    total_c = np.sum(cm[c, :])      # all true class-c samples
    correct_c = cm[c, c]            # correctly predicted class-c
    acc_c = correct_c / total_c if total_c > 0 else 0.0
    print(f"class {c}: {acc_c:.4f} ({correct_c}/{total_c})")

