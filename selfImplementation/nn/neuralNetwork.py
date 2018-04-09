"""
Created on Sun Apr 8 09:33:00 2018

@author: jercas
"""

import numpy as np

class NeuralNetwork:
	def __init__(self, layers, alpha=0.1):
		""" Initialize the list of weights matrices, then store the network architecture and learning rate. """
		self.W = []
		# A list of integers, such as [2,2,1].
		self.layers = layers
		# Learning  rate.
		self.alpha = alpha

		# Randomly sampled weights values and then normalized
		# Start looping from the index of the first layer but stop before reach the last two layers.
		for i in np.arange(0, len(layers)-2):
			# Randomly initialize a weight matrix connecting the number of nodes in each respective layer together,
			#adding an extra node as the bias node.
			w = np.random.randn(layers[i]+1, layers[i+1]+1)
			# Normalizing
			self.W.append(w / np.sqrt(layers[i]))

		# The last two layers are a special case there the input connection need a bias term but output does not.
		w = np.random.randn(layers[-2]+1, layers[-1])
		# Normalizing
		self.W.append(w / np.sqrt(layers[-2]))


	def __repr__(self):
		# Construct and return a string that represents the network architecture.
		return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))


	def sigmoid(self, x):
		# Compute and return the sigmoid activation value for a given input value.
		return 1.0 / (1 + np.exp(-x))


	def sigmoid_deriv(self, x):
		# Compute the derivative of the sigmoid function assuming that 'x' has already been passed the 'sigmoid' function.
		# sigmoid'(x) = x*(1-x)
		return x * (1-x)


	def fit(self, X, y, epochs=1000, displayUpdate=100):
		# Insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to treat the bias
		#as a trainable parameter within the weight matrix.
		X = np.c_[X, np.ones((X.shape[0]))]

		# Loop over the desired number of epochs.
		for epoch in np.arange(0, epochs):
			# Loop over each individual data point and train our network on it.
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)

			# Check to see if we should display a training update
			if epoch == 0 or (epoch+1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch={}, loss={:.8f}".format(epoch+1, loss))


	def fit_partial(self, x, y):
		# Construct list of output activations for each layer as data point flows through the network;
		#the first activation is a special case -- it's just the input feature vector itself.
		A = [np.atleast_2d(x)]

		# FEEDFORWARD:
		# Loop over the layers in the network:
		for layer in np.arange(0, len(self.W)):
			# Feedforward the activation at the current layer by taking the dot product between the activation and the
			#weight matrix -- this is called the 'net input' to the current layer
			# Zi = Θi * Xi
			net = A[layer].dot(self.W[layer])

			# Computing the 'net output' is simply applying nonlinear activation function to the net input.
			# Ai = g(Zi)
			out = self.sigmoid(net)

			# Once have the net output, add it to the list of activations
			A.append(out)

		# BACKPROPAHATION:
		# The first phase of back-propagation is to compute the difference between 'prediction'(aka the final output
		#activation in the activations list) and the true target value.
		# δlast = alast - y
		error = A[-1] - y

		# From here need to apply the chain rule and build a list of deltas 'D';
		# the first entry in the deltas is simply the error of the output layer times the derivative of activation
		#function for the output value.
		D = [error * self.sigmoid_deriv(A[-1])]

		# Loop over the layers in the reverse order, repeated until reach the first layer in the network.
		for layer in np.arange(len(A) - 2, 0, -1):
			# The delta of the current layer is equal to the delta of the 'previous layer' dotted with the weight matrix
			#of the current layer, followed by multiplying the delta by the derivative of the nonlinear activation function
			#for the activations of the current layer.
			# δi = Θi.T * δi+1 .*g'(zi)
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		# Since we looped over layers in reverse order, we need to reverse the deltas.
		D = D[::-1]

		# WEIGHT UPDATE PHASE:
		# Loop over the layers.
		for layer in np.arange(0, len(self.W)):
			# Update weights by taking the dot product of the layer activations with their respective deltas, then
			#multiplying this value by some small learning rate and adding to out weight matrix -- this is where actual
			#'learning rate' takes place.
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


	def predict(self, X, addBias=True):
		# Initialize the output prediction as the input features -- this value will be (forward) propagated through the
		#network to obtain the final prediction.
		p = np.atleast_2d(X)

		if addBias:
			# Insert a column of 1's as the last entry in the feature matrix (bias term).
			p = np.c_[p, np.ones((p.shape[0]))]

		# Loop over layers in the network.
		for layer in np.arange(0, len(self.W)):
			# Computing the output prediction is as simple as taking the dot product between the current activation value
			#'p' and the weight matrix associated with the current layer, then passing this value through a nonlinear
			#activation function.
			p = self.sigmoid(np.dot(p, self.W[layer]))

		return p


	def calculate_loss(self, X, targets):
		# Making predictions for the input data points then compute the loss.
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		return loss