"""
Created on Mon Apr 23 10:46:00 2018

@author: jercas
"""
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Initialize the model along with the default input shape to be "channels last".
		model = Sequential()
		inputShape = (height, width, depth)
		# The index of the channel dimension. Batch normalization operates over the channels, need to know which axis to normalize over.
		# Setting chanDim=-1, implies that the index of the channel dimension 'last' in the input shape(i.e., channels last ordering).
		chanDim = -1

		# If we are using "channels first", update the input shape.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			# Setting chanDim=1, implies that the index of the channel dimension 'first' in the input shape.
			chanDim = 1

		# First (CONV=>RELU=>BN)*2 =>POOL=>DO layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, name="block1_conv1"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, name="block1_conv2"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		# Keras implicitly assumes stride to be equal to the max pooling size (2,2).
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block1_pool"))
		# A node from the POOL layer will randomly disconnect from the next layer with probability of 25%.
		model.add(Dropout(0.25))

		# Second (CONV=>RELU=>BN)*2 =>POOL=>DO layer set
		model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, name="block2_conv1"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, name="block2_conv2"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		# Keras implicitly assumes stride to be equal to the max pooling size (2,2).
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool"))
		# A node from the POOL layer will randomly disconnect from the next layer with probability of 25%.
		model.add(Dropout(0.25))

		# Feed-forward architecture.
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		# FC network do not need to assign specific axis to apply normalize.
		#model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# classify.
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model