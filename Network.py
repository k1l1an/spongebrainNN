import numpy as np
from utils import activation
import pdb

class NeuralNet:

	def __init__(self,dims,activation):
		#nrinputs is size of the first layer
		self.activation = activation
		self.nr_layers = len(dims)
		self.A = [None]*(self.nr_layers-1)
		self.Z = [None]*(self.nr_layers-1)
		self.dZ = [None]*(self.nr_layers-1)
		self.W = [None]*(self.nr_layers-1)
		self.dW = [None]*(self.nr_layers-1)
		self.b = [None]*(self.nr_layers-1)
		self.db = [None]*(self.nr_layers-1)
		for i in range(0,self.nr_layers-1):
			self.W[i] = np.random.random([dims[i+1],dims[i]])*0.01
			self.b[i] = np.zeros((dims[i+1],1))


	def SGD(self,trainX,trainY,epochs=10):
		#inputs a batch with arbitrary number of training examples
		for i in range(0,epochs):
			Yhat = self.pass_forward(trainX)
			cost = self.calculate_cost(Yhat,trainY)
			self.backpropagate(cost)
			for i in range(0,self.nr_layers):
				self.W[i] -= self.dW[i]
				self.b[i] -= self.db[i]
			print("epoch nr ",str(i)," with cost ",str(cost))


	def pass_forward(self,trainX,gradient_calc = True):

		#pdb.set_trace()
		Z0 = np.dot(self.W[0],trainX)+self.b[0]
		self.Z[0] = Z0

		self.A[0] = activation(Z0,self.activation)
		
		for i in range(1,self.nr_layers-1):
			#A0 is treated as the input activation so 
			#Z0=W0*A0
			#pdb.set_trace()
			self.Z[i] = np.dot(self.W[i],self.A[i-1])+self.b[i]
			self.A[i]= activation(self.Z[i],self.activation)
		#output layer
		return self.A[-1]

	def backpropagate(self,cost,trainY):
		#calculate gradients for parameter update
		m = trainY.shape[1] #minibatchsize
		#hardcoded for L2 loss
		self.dZ[-1] = self.A[-1] -trainY
		self.dW[-1] = np.dot(self.dZ[-1],np.transpose(self.A[-2]))/m
		self.db[-1] = np.sum(self.dZ[-1],axis=1,keepdims=True)/m

		for i in range(self.nr_layers-1,0,-1):
			pdb.set_trace()
			self.dZ[i] = np.dot(np.transpose(self.W[i+1]),self.dZ[i+1])*d_activation(self.Z[i])
			self.dW[i] = np.dot(self.dZ[i],np.transpose(self.A[i-1]))
			self.db[i] = np.sum(self.dZ[i],axis=1,keepdims=True)/m

		#handle network input separately


	def calculate_cost(self,y_label,y_hat):
		return 0.5*np.mean((y_label-y_hat)**2)