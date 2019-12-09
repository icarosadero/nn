import numpy as np
import dnn_backpropagation as d

X = [np.array([1,-1,-1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([1,1,1])]
y = np.array([0,1,1,1])
#y = np.array([0,0,0,1])

dnn = d.dnn(nodes=[3,1])
dnn.m = len(y)

for epoch in range(1000):
    for obs,pair in zip(y,X):
        dnn.forward(pair)
        dnn.backward(obs)
