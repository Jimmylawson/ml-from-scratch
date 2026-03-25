import numpy as np
from sklearn.datasets import load_breast_cancer
from model import (fit_gradient_descent,accuracy,pred_prob,predict_class,precision,recall,f1_score)
import matplotlib.pyplot as plt
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
# print(f"Original means: {mu}", mu)
# print(f"Original stds: {sigma}", sigma)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
#After Standardization
# print(f"Standardized means: {X_train.mean(axis=0)}")
# print(f"Standardized stds: {X_train.std(axis=0)}")

# add bias column of one to the training and the test set to make gradient descent work
X_train = np.hstack([np.ones((X_train.shape[0], 1)),X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

#Theta initialization
theta = np.zeros(X_train.shape[1])
alpha = 0.01
num_iters = 2000
#fit gradient descent
theta, cost_history = fit_gradient_descent(X_train, Y_train, theta,num_iters, alpha)


# print(f"First cost: {cost_history[0]}")      # Should be high
# print(f"Last cost: {cost_history[-1]}")      # Should be low
# print(f"Cost improvement: {cost_history[0] - cost_history[-1]}")


#accuracy
train_acc = accuracy(X_train,Y_train,theta)
test_acc = accuracy(X_test,Y_test,theta)

#print(f"Train accuracy: {train_acc}") # this  i got 98# training accuracy
#print(f"Test accuracy: {test_acc}") # this i got 100% training accuracy

#probability
# train_prob = pred_prob(X_train,theta)
# test_prob = pred_prob(X_test,theta)
# print(f"Train probability: {train_prob[-1]}")
# print(f"Test probability: {test_prob[-1]}")

#print confusion matrix which tells us how our classifier performs
# print("Confusion matrix:")
# for t in [0.3,0.5,0.7]:
#     print(f"Threshold: {t}\n")
#     y_pred_test = predict_class(X_test,theta,t) #gives your model's prediction (0 or 1)
#     tp = np.sum((y_pred_test == 1) & (Y_test == 1))  # True Positive: model predicted 1 and actual = 1
#     tn = np.sum((y_pred_test == 0) & (Y_test == 0))  # True Negative: model predicted  0 and actual 0
#     fp = np.sum((y_pred_test == 1) & (Y_test == 0))  # False positive: Model predicted 1 but actual = 0
#     fn = np.sum((y_pred_test == 0) & (Y_test == 1))  # False negative: Model predicted 0 but actual = 1
#     # let you know how many correct predictions
#     # how many mistakes
#     # what type of mistakes
#     print(f"TN={tn}, FP={fp}")
#     print(f"FN={fn}, TP={tp}")
#     # classification metrics
#     print(f"Precision: {precision(tp, fp)}")
#     print(f"Recall: {recall(tp, fn)}")
#     print(f"F1 Score: {f1_score(tp, fp, fn)}\n")



#Convergence plot(cost vs iterations)
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Logistic Cost")
plt.title("Logistic GD Convergence")
plt.grid(True)
plt.savefig("logistic_gd_convergence.png")
plt.show()
#Convergence is  where theta stop changing significantly