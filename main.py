# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:28:45 2020

@author: kilia
"""


import gzip
import matplotlib.pyplot as plt
import numpy as np
from Network import NeuralNet
from utils import to_one_hot 
import pdb


#load training data
np.random.seed(1)

image_size = 28
num_images = 5


#extract images
with gzip.open('train-images-idx3-ubyte.gz','r') as f:
	f.read(16)
	buf = f.read(image_size * image_size * num_images)
	data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	data = data.reshape(num_images,image_size, image_size)
	data = data.transpose(1,2,0)
	data = data.reshape(-1,5)
	dataX = data/255

#extract labels
with gzip.open('train-labels-idx1-ubyte.gz','r') as g:
	g.read(8)
	buf = g.read(num_images)
	data = np.frombuffer(buf,dtype=np.uint8)
	dataY = to_one_hot(data,10)




nn = NeuralNet([784,100,99,88,77,10],'sigmoid')

y_hat = nn.pass_forward(dataX)
cost = nn.calculate_cost(dataY,y_hat)

nn.backpropagate(cost,dataY)





#plt.imshow(image)
#plt.show()

