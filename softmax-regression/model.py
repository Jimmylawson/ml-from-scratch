
import numpy as np


def predict(X,theta):
    return X @ theta


# keepdims=True → keeps shape (m,1) instead of (m,)
# → makes broadcasting safe and correct

def softmax(X, theta):
    logits = predict(X,theta)
    logits = logits -  np.max(logits, axis = 1,keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis = 1,keepdims=True)

#take log → penalize wrong predictions
#take negative → turn into loss
#take mean → average over all examples

def cross_entropy_loss(X, y, theta):
    m = len(y)
    probs = softmax(X, theta) # (m, k)
    #epsiln ensures the
    # cost function never encounter log(0)
    # or log(negative) by keeping the predictions
    # safely away from 0 and 1 boundaries!
    epsilon = 1e-15

    probs = np.clip(probs, epsilon, 1 - epsilon)
    correct_class_probs = probs[np.arange(m), y] # the true-class probability from each row.
    return np.mean(-np.log(correct_class_probs))



def softmax_gradient(X,y, theta):
    m = len(y)
    probs  = softmax(X, theta)
    K = theta.shape[1]
    y_onehot = np.eye(K)[y]  # (m, K)
    # create identity matrix (K x K), where each row represents a class
    # use y as indices to select rows → converts labels into one-hot vectors

    error = probs - y_onehot #(m, k)
    gradient = (1/m) * ( X.T @ error) #(n,k)
    return gradient

def fit_softmax_gd(X,y,theta, alpha, num_iters,log_every=None):
    cost_history = []
    for i in range(num_iters):
        gradient = softmax_gradient(X,y,theta)
        theta = theta - alpha * gradient
        cost = cross_entropy_loss(X, y, theta)
        cost_history.append(cost)

        #log_every controls how often to print training progress.
        if log_every is not None and (i % log_every == 0 or i == num_iters - 1):
            print(f"iter {i:4d} | loss {cost:.6f}")

    return theta, cost_history


def predict_class(X,theta):
    probs = softmax(X,theta)
    return np.argmax(probs, axis=1)

def accuracy(X,y, theta):
    preds = predict_class(X,theta)
    return np.mean(preds == y)
