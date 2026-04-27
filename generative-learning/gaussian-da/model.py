import numpy as np



def class_prior(y_train):
    classes = np.unique(y_train)
    return np.array([np.mean(y_train == k) for k in classes]), classes

def mean_vector(X_train,y_train,classes):
    return np.array([np.mean(X_train[y_train == k],axis= 0) for k in classes])
def mean_vector_zero(X_train,y_train):
    return np.mean(X_train[y_train == 0], axis=0)

def mean_vector_one(X_train,y_train):
    return np.mean(X_train[y_train == 1], axis=0)

# #for binary GDA
# def covariance_matrix(X_train,y_train):
#     mu0 = mean_vector_zero(X_train,y_train)
#     mu1 = mean_vector_one(X_train,y_train)
#     m = len(y_train)
#     n_features = X_train.shape[1]
#     # this will create a matrix of size (n_features, n_features)
#     # where covariance compares every feature with every other feature
#     sigma = np.zeros((n_features, n_features))
#
#     for i in range(m):
#         if y_train[i] == 0:
#             diff = X_train[i] - mu0
#         else:
#             diff = X_train[i] - mu1
#
#         diff = diff.reshape(-1,1) #reshape vector a shape example like (4,) so we can do the outer product
#         sigma += diff @ diff.T
#     return sigma / m

# for multiclass GDA
def covariance_matrix(X_train,y_train, classes,  mus):
    m = len(y_train)
    n_features = X_train.shape[1]
    sigma = np.zeros([n_features, n_features])

    for i in range(m):
        class_index = np.where(classes == y_train[i])[0][0]
        diff = X_train[i] - mus[class_index]
        diff = diff.reshape(-1,1)
        sigma += diff @ diff.T
    return sigma / m


#the log likelihood here and the prediction_one  are for binary GDA  but for multiclass GDA is different
# gaussian_log_likelihood can be use for multiclass GDA
def gaussian_log_likelihood(x, mu, sigma):
    d = len(x)
    diff =  x - mu
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    return (
        -0.5 * diff.T @ inv_sigma @ diff
        -0.5 * np.log(det_sigma)
        -0.5 * d * np.log(2 * np.pi)
    )


# def prediction_one(x,phi_y, mu0, mu1, sigma):
#     score0 = np.log(1 - phi_y) +  gaussian_maximum_likelihood(x,mu0,sigma)
#     score1 = np.log(phi_y) + gaussian_maximum_likelihood(x,mu1,sigma)
#
#     scores = np.array([score0,score1])
#     return np.argmax(scores)

def predict_one(x,prior,classes, mus, sigma):
    scores = []

    for k_idx, k in enumerate(classes):
        score = np.log(prior[k_idx]) +  gaussian_log_likelihood(x,mus[k_idx], sigma)
        scores.append(score)


    return classes[np.argmax(scores)]





