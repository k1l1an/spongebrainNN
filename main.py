# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:28:45 2020

@author: kilia
"""


import gzip
import matplotlib.pyplot as plt
import numpy as np


#load training data

f = gzip.open('train-images-idx3-ubyte.gz','r')
g = gzip.open('train-labels-idx1-ubyte.gz','r')
image_size = 28
num_images = 5

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = np.asarray(data[4]).squeeze()
plt.imshow(image)
plt.show()


if __name__ == "__main__":
    print('main')