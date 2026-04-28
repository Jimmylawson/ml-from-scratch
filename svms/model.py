
import numpy as np

def prediction_score(X, w, b):
    # X @ w linear combination  and + b is the bias/intercept which is to shift the boundary
    return  X @ w + b


def soft_margin_sv_loss(X,y,w,b,C):
    margins = y * prediction_score(X, w, b)
    losses = np.maximum(0, 1 - margins)
    return 0.5 * np.dot(w,w) + C * np.mean(losses)
