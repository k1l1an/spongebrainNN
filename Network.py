import numpy as np
from utils import activation, d_activation
import pdb


class NeuralNet:

	def __init__(self, dims, activation):
		# nrinputs is size of the first layer
		self.activation = activation
		self.nr_layers = len(dims) - 1
		self.A = [None] * (self.nr_layers)
		self.Z = [None] * (self.nr_layers)
		self.dZ = [None] * (self.nr_layers)
		self.W = [None] * (self.nr_layers)
		self.dW = [None] * (self.nr_layers)
		self.b = [None] * (self.nr_layers)
		self.db = [None] * (self.nr_layers)
		for i in range(0, self.nr_layers):
			self.W[i] = np.random.random([dims[i + 1], dims[i]]) * 0.01
			self.b[i] = np.zeros((dims[i + 1], 1))

	def SGD(self, trainX, trainY, epochs=10):
		# inputs a batch with arbitrary number of training examples
		for epoch in range(0, epochs):
			Yhat = self.feedforward(trainX)
			cost = self.calculate_cost(Yhat, trainY)
			self.backpropagate(cost,trainX,trainY)
			for i in range(0, self.nr_layers):
				self.W[i] -= self.dW[i]
				self.b[i] -= self.db[i]
			print("epoch nr ", str(epoch), " with cost ", str(cost))

	def feedforward(self, trainX, gradient_calc=True):

		# pdb.set_trace()
		Z0 = np.dot(self.W[0], trainX) + self.b[0]
		self.Z[0] = Z0

		self.A[0] = activation(Z0, self.activation)

		for i in range(1, self.nr_layers):
			# A0 is treated as the input activation so
			# Z0=W0*A0
			# pdb.set_trace()
			self.Z[i] = np.dot(self.W[i], self.A[i - 1]) + self.b[i]
			self.A[i] = activation(self.Z[i], self.activation)
		# output layer
		return self.A[-1]

	def backpropagate(self, cost, trainX,trainY):
		# calculate gradients for parameter update
		m = trainY.shape[1]  # minibatchsize
		# hardcoded for L2 loss
		self.dZ[-1] = self.A[-1] - trainY
		self.dW[-1] = np.dot(self.dZ[-1], np.transpose(self.A[-2])) / m
		self.db[-1] = np.sum(self.dZ[-1], axis=1, keepdims=True) / m

		#for 2 layers this is not executed at all
		for i in range(self.nr_layers - 1, 0, -1):
			try:
				self.dZ[i - 1] = np.dot(np.transpose(self.W[i]), self.dZ[i]) * d_activation(self.Z[i-1])
			except IndexError:
				pass
			self.dW[i] = np.dot(self.dZ[i], np.transpose(self.A[i - 1])) / m
			self.db[i] = np.sum(self.dZ[i], axis=1, keepdims=True) / m

	# handle network input separately

		self.dW[0] = np.dot(self.dZ[0], np.transpose(trainX)) / m
		self.db[0] = np.sum(self.dZ[0], axis=1, keepdims=True) / m


	def calculate_cost(self, Y_label, Y_hat):
		return 0.5 * np.mean((Y_label - Y_hat) ** 2)
