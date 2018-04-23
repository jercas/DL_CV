"""
Created on Fri Apr 20 16:51:00 2018

@author: jercas
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# First set of CNOV=>RELU=>POOL
		model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		# Second set of CNOV=>RELU=>POOL
		model.add(Conv2D(50, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# FC layers set
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model 