"""
Created on Mon Apr 23 16:35:00 2018

@author: jercas
"""
import numpy as np

def stepBased_decay(epoch):
	# Initialize the base initial learning rate α, drop factor and epochs to drop every set of epochs.
	initialAlpha = 0.01
	# Drop learning rate by a factor of 0.25 every 5 epochs.
	factor = 0.25
	dropEvery = 5

	# Compute learning rate for the current epoch.
	alpha = initialAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
	return float(alpha)