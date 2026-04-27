import numpy as np



def class_prior(y_train):
    return np.mean(y_train)

def mean_vector_zero(X_train,y_train):
    return np.mean(X_train[y_train == 0], axis=0)

def mean_vector_one(X_train,y_train):
    return np.mean(X_train[y_train == 1], axis=0)

def covariance_matrix(X_train,y_train):
    mu0 = mean_vector_zero(X_train,y_train)
    mu1 = mean_vector_one(X_train,y_train)
    m = len(y_train)
    n_features = X_train.shape[1]
    # this will create a matrix of size (n_features, n_features)
    # where covariance compares every feature with every other feature
    sigma = np.zeros((n_features, n_features))

    for i in range(m):
        if y_train[i] == 0:
            diff = X_train[i] - mu0
        else:
            diff = X_train[i] - mu1

        diff = diff.reshape(-1,1) #reshape vector a shape example like (4,) so we can do the outer product
        sigma += diff @ diff.T
    return sigma / m


def gaussian_maximum_likelihood(x, mu, sigma):
    d = len(x)
    diff =  x - mu
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    return (
        -0.5 * diff.T @ inv_sigma @ diff
        -0.5 * np.log(det_sigma)
        -0.5 * d * np.log(2 * np.pi)
    )


def prediction_one(x,phi_y, mu0, mu1, sigma):
    score0 = np.log(1 - phi_y) * gaussian_maximum_likelihood(x,mu0,sigma)
    score1 = np.log(phi_y) * gaussian_maximum_likelihood(x,mu1,sigma)

    scores = np.array([score0,score1])
    return np.argmax(scores)




