import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	"""
	data: (n, d)
	"""

	eps = 1e-10
	
	mean = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)

	maxf = np.max(data, axis = 0)
	minf = np.min(data, axis = 0)

	standardized_data = (data - mean) / (std + eps)
	normalized_data = (data - minf) / (maxf - minf + eps)

	return np.round(standardized_data, 4), np.round(normalized_data, 4)