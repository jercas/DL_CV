"""
Created on Sat Mar 31 13:55:00 2018

@author: jercas
"""

from nn.perceptron import Perceptron
import numpy as np

# Construct the OR dataset.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Define perceptron and train it.
print("[INFO] training percetron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# Evaluate trained model.
print("[INFO] testing percrtron...")

for (x, target) in zip(X, y):
	pred = p.predict(x)
	print("[INFO] data={}, ground_true={}, pred={}".format(x, target[0], pred))