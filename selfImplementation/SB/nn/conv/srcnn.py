"""
Created on Tue Jun 19 10:32:00 2018

@author: jercas
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
import keras.backend as K

class SRCNN:
	@staticmethod
	def build(width, height, depth):
		# Initialize the model along with the input shape to be "channels last" and the channels dimension itself.
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# If channels order is "channels first", modify the input shape and channels dimension.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Construct the model architecture.
		# The entire SRCNN architecture consists of three CONV => RELU layers with *no zero-padding*.
		model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal", input_shape=inputShape))
		model.add(Activation("relu"))

		model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
		model.add(Activation("relu"))

		model.add(Conv2D(depth, (5, 5), kernel_initializer="he_normal"))
		model.add(Activation("relu"))

		# Return the constructed model.
		model.summary()
		return model