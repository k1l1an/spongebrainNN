import numpy as np
import pdb
def to_one_hot(x,dim):
	mat = np.zeros((dim,len(x)))
	mat[x,range(0,len(x))] = 1
	return mat

def activation(X,fct="tanh"):
	if fct == "sigmoid":
		return 1/(1+np.exp(-X))
	elif fct == "tanh":
		return np.tanh(X)

def d_activation(X,fct='tanh'):
	if fct == 'sigmoid':
		sig = activation(X,'sigmoid')
		return sig*(1-sig)
	if fct == 'tanh':
		th = activation(X,fct='tanh')
		return 1-th**2
