import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	
	W = weights.reshape(-1, 1) #(d, 1)
	z = X @ W + bias #(n, 1)

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	
	probs = sigmoid(z).reshape(-1)

	return (probs >= 0.5).astype(int)
