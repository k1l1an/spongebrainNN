import numpy as np
def vectorize_label(x):
	vec = np.zeros((10,1))
	vec[x] = 1
	return vec