import numpy as np

class NeuralNet:

	def __init__(self,dims,activation):
		#nrinputs is size of the first layer
		self.nr_layers = len(dims)
		self.A = []
		self.Z = []
		self.W = []
		self.dW = []
		self.b = []
		self.db = []
		for i in range(0,self.nr_layers-1):
			self.W.append(np.random.random([dims[i+1],dims[i]])*0.01)
			self.b.append(np.zeros((dims[i+1],1)))


	def SGD(self,trainX,trainY,epochs=10):

		for i in range(0,epochs):
			Yhat = self.pass_forward(trainX)
			cost = self.calculate_cost(Yhat,trainY)
			self.backpropagate(cost)
			for i in range(0,self.nr_layers):
				self.W[i] -= self.dW[i]
				self.b[i] -= self.db[i]
			print("epoch nr ",str(i)," with cost ",str(cost))


		return
	def activation(self,X,fct="tanh"):
		if fct == "sigmoid":
			return 1/(1+np.exp(-X))
		elif fct == "tanh":
			return np.tanh(X)

	def pass_forward(self,train_X,gradient_calc = True):

		Z1 = np.dot(self.W[0],train_X)+self.b[0]
		self.Z.append(Z1)

		self.A.append(self.activation(Z1))
		
		for i in range(1,self.nr_layers-1):
			Z_curr = np.dot(self.W[i],self.A[i-1])+self.b[i]
			self.Z.append(Z_curr)
			self.A.append(self.activation(Z_curr))
		#output layer
		return self.A[-1]

	def backpropagate(self,cost):
		#calculate gradients for parameter update
		for i in range(self.nr_layers,0,-1):
			self.dZ[i]	

			self.db[i] = 
		return

	def calculate_cost(self,X,Y):
		return np.mean((X-Y)**2)