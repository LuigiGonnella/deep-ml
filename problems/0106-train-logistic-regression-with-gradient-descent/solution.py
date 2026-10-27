import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""

	losses = []
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	n_samples, n_features = X.shape
	#first dim is BIAS
	#W = np.random.randn(1, n_features + 1) * np.sqrt((1 / (n_features + 1))) #xavier init including bias
	W = np.zeros((1, n_features + 1))
	y = y.reshape(-1, 1) #(n_samples, 1)
	X = np.concatenate([np.ones((n_samples, 1)), X], axis=1) #first column multiplies the bias, so the weighted su will have "b + w1x1 + ... + w_dx_d"
	for _ in range(iterations):
		z = X @ W.T #(n_samples, 1)
		pred = sigmoid(z) #(n_samples, 1) as PROBS

		loss = - np.sum(y * np.log(pred) + (1 - y) * np.log((1 - pred))) #sum BCE, not mean
		losses.append(loss.reshape(-1).item())

		dW = (pred - y).T @ X #(1, n_features + 1)

		W -= learning_rate * dW
	
	return (W.reshape(-1).tolist(), losses)




