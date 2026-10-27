import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    
    weights: (d, )
    """

    n_samples = len(batch_indices)
    x = X[batch_indices] #(n_samples, n_features)
    y = y.reshape(-1, 1)[batch_indices] #(n_samples, 1)
    weights = weights.reshape(1, -1) #(1, n_features)
    pred = x @ weights.T + bias #(n_samples, 1)

    err = pred - y #(n_samples, 1)

    loss = (1 / n_samples) * np.sum(err ** 2) #()

    dW = (2 / n_samples) * err.T @ x #(1, n_features)
    db = (2 / n_samples) * np.sum(err) #()

    weights -= lr * dW
    bias -= lr * db

    W = weights.reshape((-1,))
    bias = bias.reshape((-1,))

    return np.concatenate([W, bias], axis=0)
