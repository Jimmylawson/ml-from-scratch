
import numpy as np

def prediction_score(X, w, b):
    # X @ w linear combination  and + b is the bias/intercept which is to shift the boundary
    return  X @ w + b

#how bad the model is
def soft_margin_svm_loss(X,y,w,b,C):
    margins = y * prediction_score(X, w, b)
    losses = np.maximum(0, 1 - margins)
    return 0.5 * np.dot(w,w) + C * np.mean(losses)

# how to change the model to make it better
def soft_margin_svm_gradient(X,y,w,b,C):
    margins = y * prediction_score(X, w, b)
    violating  = margins < 1
    dw = w.copy()
    db  = 0

    if np.any(violating):
        dw -= C * np.mean(y[violating, None] * X[violating], axis=0)
        db -= C * np.mean(y[violating])

    return dw, db


def fit_svm(X, y, alpha, C, num_iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = []

    for i in range(num_iters):
        dw, db = soft_margin_svm_gradient(X, y, w, b, C)

        w = w - alpha * dw
        b = b - alpha * db

        loss = soft_margin_svm_loss(X, y, w, b, C)
        loss_history.append(loss)

    return w, b, loss_history

#just checks which side of that boundary each point is on.
def predict_class(X,w,b):
    scores = prediction_score(X, w, b)
    return np.where(scores >= 0, 1, -1)