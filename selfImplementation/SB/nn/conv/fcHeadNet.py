"""
Created on Sun May 13 11:11:00 2018

@author: jercas
"""

from keras.layers.core import Dropout, Dense, Flatten

class FCHeadNet:
	@staticmethod
	def build(baseModel, classes, D):
		"""
			Build a new CNN architecture based on a state-of-art CNN(i.e., transform learning from it)
			Parameters:
				baseModel: The body of no-top(be clipped the FC layers at the head of architecture) original network.
				classes: The categories of the dataset which you want to transform learning on.
				D: The number of nodes in the FC layer.
			Return:
				newModel: The new FC head architecture.
		"""

		# INPUT FROM MAIN BODY => FC => RELU => DROPOUT => FC => SOFTMAX => OUTPUT PROBABILITY
		originalModel = baseModel.output
		headModel = Flatten(name="flatten")(originalModel)
		headModel = Dense(D, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(classes, activation="softmax")(headModel)

		return headModel