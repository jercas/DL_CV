"""
Created on Mon Apr 23 10:46:00 2018

@author: jercas
"""
from keras.utils import plot_model

def modelSaver(model, modelname, filePath):
	# Save whole trained model.
	model.save("{}/{}.hdf5".format(filePath, modelname))
	# Just save weights.
	model.save_weights("{}/{}.hdf5".format(filePath, modelname))
	# Save the visualization image of network architecture.
	plot_model(model, to_file="{}/{}.png".format(filePath, modelname))