
import numpy as np


def predict(X,theta):
    return X @ theta


# keepdims=True → keeps shape (m,1) instead of (m,)
# → makes broadcasting safe and correct

def softmax(X, theta):
    logits = predict(X,theta)
    # prevent numerical overflow.
    #without shifting some large values of exp(logits) will be infinite
    #keepdims maintain the 2D shape for proper broadcasting operations!
    #Shape explanation:
    #Input: exp_logits with shape (m_samples, k_classes)
    #np.sum(..., axis=1): Sums across columns → shape (m_samples, 1)
    #Division: (m,k) / (m,1) → shape (m,k) (broadcasting)
    # Output: (m_samples, k_classes) probabilities
    # Single value vs multi-class:
    # Single value: Logistic regression (2 classes)
    # Multi values: Softmax regression (3+ classes)

    logits = logits -  np.max(logits, axis = 1,keepdims=True)
    exp_logits = np.exp(logits)
    # will return the probability distribution across all classes.
    return exp_logits / np.sum(exp_logits, axis = 1,keepdims=True)


# What cross-entropy does:
#
# Penalizes wrong predictions: -log(probability)
# High correct probability → Low loss: -log(0.9) = 0.105
# Low correct probability → High loss: -log(0.1) = 2.303
# Average across all samples
# Why this works:
#
# Good model: High confidence in correct classes → low loss
# Bad model: Low confidence in correct classes → high loss
# Training: Minimize loss → increase confidence in correct predictions
# So cross-entropy rewards correct predictions and penalizes incorrect ones!

def cross_entropy_loss(X, y, theta):
    m = len(y)
    probs = softmax(X, theta) # (m, k)
    #epsiln ensures the
    # cost function never encounter log(0)
    # or log(negative) by keeping the predictions
    # safely away from 0 and 1 boundaries!
    epsilon = 1e-15

    probs = np.clip(probs, epsilon, 1 - epsilon)
    # # For 3 samples:
    # probs = [[0.1, 0.7, 0.2],  # Sample 0: actual class 1
    #          [0.6, 0.3, 0.1],  # Sample 1: actual class 0
    #          [0.05, 0.2, 0.75]]  # Sample 2: actual class 2
    #
    # y = [1, 0, 2]  # Correct classes
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
        # if log_every is not None and (i % log_every == 0 or i == num_iters - 1):
        #     print(f"iter {i:4d} | loss {cost:.6f}")

    return theta, cost_history

# # Input: One digit image
# probs = [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.85, 0.03, 0.02]  # 10 classes
# #          [0,   1,   2,   3,   4,   5,   6,   7,   8,   9]
#
# # Output: Predicted class
# prediction = np.argmax(probs) → 7  # Highest probability (85%) for digit 7

# Purpose:
# Training: Evaluate model performance
# Inference: Make actual predictions on new data
# Evaluation: Compare predictions to actual labels
# So predict_class converts probability distributions to discrete class labels!
def predict_class(X,theta):
    #Return the probs for all the classes
    probs = softmax(X,theta)
    #find the index of the highest probability
    return np.argmax(probs, axis=1)

# High probability doesn't guarantee high accuracy:
# probabilities = [0.95, 0.92, 0.88, 0.93, 0.90]  # All high confidence
# predictions  = [7, 1, 7, 1, 7]                # But might be wrong!
# actual      = [7, 9, 7, 1, 7]                # Some wrong
# accuracy = 3/5 = 60%  # Despite high confidence!

#Accuracy measure overall performance not the individual class performance
def accuracy(X,y, theta):
    preds = predict_class(X,theta)
    return np.mean(preds == y)
