"""
Created on Wed May 30 19:33:00 2018

@author: jercas
"""

def polynomial_decay(epoch):
	# Initialize the maximum number of epochs, initial learning rate α and the power of the polynomial.
	MAX_EPOCHS = 70
	INIT_LR = 5e-3
	POWER = 1.0

	# Compute the updated learning rate based on polynomial decay.
	alpha = INIT_LR * (1 - (epoch / float(MAX_EPOCHS))) ** POWER

	# Return the updated learning rate α.
	return alpha