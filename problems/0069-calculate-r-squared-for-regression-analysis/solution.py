
import numpy as np

def r_squared(y_true, y_pred):
	
	true_mean = np.mean(y_true)

	res_sum_square= np.sum((y_true - y_pred) ** 2)
	tot_sum_square = np.sum((y_true - true_mean) ** 2)

	return 1 - res_sum_square / (tot_sum_square + 1e-10)
