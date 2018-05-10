"""
Created on Thu May 10 13:54:00 2018

@author: jercas
"""
import numpy as np

def rank5_acc(preds, labels):
	"""
		Compute the rank-1 and rank-5 accuracy of the predictions.
		Parameters:
			preds: model's output predictions.
			labels: the ground-truth labels of the testset.
		Returns:
			 rank1: rank-1 accuracy.
			 rank5: rank-5 accuracy.
	"""
	# Initialize the rank-1 and rank-5 accuracy.
	rank1 = 0
	rank5 = 0

	# Loop over the predictions and "ground-truth labels".
	for (pred, gtlab) in zip(preds, labels):
		# Sort the predictions probabilities by their index in descending order([::-1]) so that the more confident guesses
		#are at the front of the list.
		pred = np.argsort(pred)[::-1]

		# Check if the "ground-truth label" is in the top-5 probabilities, if so consider this prediction as correct.
		if gtlab in pred[:5]:
			rank5 += 1

		# Check to see if the "ground-truth label" is the highest probability.
		if gtlab == pred[0]:
			rank1 += 1

	# Compute the final rank-1 and rank-5 accuracies.
	rank1 /= float(len(preds))
	rank5 /= float(len(preds))

	return (rank1, rank5)