import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:   
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    for _ in range(iterations):
        pred = X @ theta #(m, 1)

        err = pred - y #(m, 1)
        loss = 1 / (2 * m) * np.sum(err ** 2) #()
        
        dtheta = (1 / m) * X.T @ err #(n, 1) --> 1 or 2 depending on loss, if simple mean, then 2 / m, if th mean has 2 in denominator then it simplifies and we have just 1 / m in gradient

        theta -= alpha * dtheta

    return theta.flatten()