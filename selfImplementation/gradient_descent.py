"""
Created on Mon Mar 26 16:43:56 2018

@author: jercas
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# A function used to create "blobs" of normally distributed data points, a handy function when testing or implementing our own models from scratch.
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
	'vanilla'/standard gradient descent algorithm.
"""

def sigmoid(x):
	""" Compute the sigmoid activation value for a given input."""
	return 1.0 / (1+np.exp(-x))


def predict(X, W):
	# Take the dot product between features and weight matrix.
	preds = sigmoid(X.dot(W))

	# Apply a step function to threshold the outputs to binary class labels.
	preds[preds <= 0.5] = 0
	preds[preds > 0.5] = 1
	return preds

def next_batch(X, y, batchSize):
	# Loop over dataset 'X' in mini-batches, yielding a tuple of the current batched data and labels.
	for i in np.arange(0, X.shape[0], batchSize):
		yield(X[i:i+batchSize], y[i:i+batchSize])

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--epochs",     type=float, default=100,  help="# of epochs")
	ap.add_argument("-a", "--alpha",      type=float, default=0.01, help="learning rate")
	ap.add_argument("-b", "--batch_size", type=int,  help="size of SGD mini-batches")
	args = vars(ap.parse_args())

	# Generate a 2-class classification problem with 1,000 data points, where each data point is a 2D feature vector.
	(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
	y = y.reshape((y.shape[0], 1))

	# Insert a column of 1's as the last entry in the feature matrix -- as the bias dimension.
	# np.c_[] -- Translates slice objects to concatenation along the second axis.
	X = np.c_[X, np.ones((X.shape[0]))]

	# Partition the data into training and testing splits using 50% of the data for training and the remaining 50% for testing.
	(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

	# Initialize our weight matrix and list of losses.
	print("[INFO] training...")
	W = np.random.randn(X.shape[1], 1)

	# A list that keep track of losses after each epoch for plot loss later.
	losses = []

	if args["batch_size"]:
		for epoch in np.arange(0, args["epochs"]):
			# Initialize the total loss for the epoch.
			epochLoss = []
			# Loop over data in batches.
			for (batchX, batchY) in next_batch(X, y ,args["batch_size"]):
				# Take the dot product between current batch of features and the weight matrix, then pass this value through
				#activation function
				preds = sigmoid(batchX.dot(W))
				error = preds - batchY
				epochLoss.append(np.sum(np.square(error)))

				gradient = batchX.T.dot(error)
				W += -args["alpha"] * gradient

			# Update loss history by taking the average loss across all batches.
			loss = np.average(epochLoss)
			losses.append(loss)

			# Check to see if an update should be displayed.
			if epoch == 0 or (epoch + 1) % 5 == 0:
				print("[INFO] epoch={} , loss={:.7f}".format(int(epoch + 1), loss))

	else:
		# Loop over the desired number of epochs.
		for epoch in np.arange(0, args["epochs"]):
			# Take the dot product between features 'X' and the weight matrix 'W', then pass this value through our sigmoid
			#activation function, thereby giving predictions on the dataset.
			preds = sigmoid(trainX.dot(W))

			# Computing the 'error' which is the difference between predictions and the ground-true label values.
			error = preds - trainY
			# MSE(mean square error) of each data point.
			loss = np.sum(np.square(error))
			losses.append(loss)

			# The gradient descent update is the dot product between features and the error of the predictions.
			gradient = trainX.T.dot(error)
			# In the update stage, all we need to do is "nudge" the weight matrix in the negative direction of the gradient(
			#hence the term 'gradient descent' by taking a small step towards a set of 'more optimal' parameters.
			W += -args["alpha"] * gradient

			# Check to see if an update should be displayed.
			if epoch == 0 or (epoch + 1) % 5 == 0:
				print("[INFO] epoch={} , loss={:.7f}".format(int(epoch+1), loss))

	# Evaluate model
	print("[INFO] evaluating...")
	preds = predict(testX, W)
	print(classification_report(testY, preds))

	# Plot the testing classification data
	plt.style.use("ggplot")
	plt.figure()
	plt.title("Data")
	colors = testY.reshape(1, testY.shape[0])[0]
	plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=colors, s=30)
	plt.show()

	# Construct a figure that plots the loss over time.
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, args["epochs"]), losses)
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.show()


if __name__ == "__main__":
	main()