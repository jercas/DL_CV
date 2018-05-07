"""
Created on Sun Apr 8 16:38:00 2018

@author: jercas
"""

import numpy as np
from nn.neuralNetwork import NeuralNetwork

# Construct the XOR dataset.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Define a 2-2-1 neural network and train it.
# With one hidden layer, model will be able to learning nonlinear function.
nn = NeuralNetwork([2,2,1], alpha=0.5)
# With no hidden layer, aka a model that just as a perceptron isn't capable to learning the nonlinear pattern.
#nn = NeuralNetwork([2,1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# Loop over the XOR data points and make prediction.
for (x, target) in zip(X, y):
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("\n[INFO] data={}, ground-truth={}, pred={:.6f}, step={}".format(x, target[0], pred, step))