"""
Created on Thu Mar 29 11:09:00 2018

@author: jercas
"""

import numpy as np

class Perceptron():
	def __init__(self, N , alpha=0.1):
		"""
			N: The number of columns in input feature vectors.
			alpha: Learning rate.
		"""
		# Initialize the weight matrix W with random values sampled from a normal(Gaussian) distribution with zero mean
		# and unit variance.
		self.W = np.random.randn(N+1) / np.sqrt(N)
		# Store the learning rate.
		self.alpha = alpha


	def step(self, x):
		return 1 if x>0 else 0


	def fit(self, X, y, epochs=10):
		# Add bias term
		X = np.c_[X, np.ones((X.shape[0]))]
		# Loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# Loop over each individual data point.
			for (x, target) in zip(X, y):
				# Take the dot product between the input features and the weight matrix, then pass this value through
				#the step function to obtain the prediction.
				p = self.step(np.dot(x, self.W))

				# Only perform a weight update if prediction does not match the target.
				if p != target:
					# Determine the error
					error = p - target
					# Update the weight matrix
					self.W += -self.alpha * error * x


	def predict(self, X, addBias=True):
		# Ensure to be classified input is a matrix.
		X = np.atleast_2d(X)

		# Check to see if the bias column should be added/
		if addBias:
			# Insert a column of 1's as the last entry in the feature matrix(bias).
			X = np.c_[X, np.ones((X.shape[0]))]

		return self.step(np.dot(X, self.W))
