import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	#solve in closed form -->  THETA = (X.T @ X) ^ (-1) @ X.T * Y
	X, y = np.array(X), np.array(y)
	theta = np.linalg.solve(np.transpose(X) @ X, np.transpose(X) @ y)

	return theta.tolist()