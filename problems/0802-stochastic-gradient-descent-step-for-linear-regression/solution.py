import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """

    n_samples, n_features = X.shape
    weights = weights.reshape(-1, 1) #(n_features, 1)
    for i in range(n_iter):
        idx = i % n_samples

        x = X[idx].reshape(1, -1) #(1, n_features)
        newy = y[idx].reshape(1, -1) #(1, 1)

        pred = x @ weights #no bias (n_samples, 1)

        err = pred - newy #(n_samples, 1)
        loss = np.sum(err ** 2) #()

        dweights = 2 * x.T @ err #(n_features, 1)

        weights -= learning_rate * dweights #(n_features, 1)
    
    return weights.flatten().tolist() #len = n_features
