"""
Created on Tue May 22 12:31:00 2018

@author: jercas
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout

class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		"""
		Model builder.
		Parameter:
			width: The width dimension of image/number of horizontal pixels.
			height: The height dimension of image/number of vertical pixels.
			depth: The number of image channels.
			classes:
			finalAct:  The optional argument, finalAct (with a default value of "softmax") will be utilized at the end of the network architecture.
					   Changing this value from 'softmax' to 'dsigmoid' will enable us to perform 'multi-label classification' with Keras.
					   Control whether we are performing 'simple classification' or 'multi-class classification'.
		Return:
			model: The constructed network architecture.
		"""
		# Initialize the model along with the input shape to be "channels last" and the channels dimension itself.
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# If channels order is "channels first", modify the input shape and channels dimension.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# First CONV block, CONV => RELU => POOL.
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, name="block_1--CONV_1"))
		model.add(Activation("relu", name="block_1--ACT_relu_1"))
		model.add(BatchNormalization(axis=chanDim, name="block_1--BN_1"))
		model.add(MaxPooling2D(pool_size=(3, 3), name="block_1--POOL_max"))
		model.add(Dropout(0.25, name="block_1--DO"))

		# Second CONV block, (CONV => RELU)*2 => POOL.
		model.add(Conv2D(64, (3, 3), padding="same", name="block_2--CONV_1"))
		model.add(Activation("relu", name="block_2--ACT_relu_1"))
		model.add(BatchNormalization(axis=chanDim, name="block_2--BN_1"))
		model.add(Conv2D(64, (3, 3), padding="same", name="block_2--CONV_2"))
		model.add(Activation("relu", name="block_2--ACT_relu_2"))
		model.add(BatchNormalization(axis=chanDim, name="block_2--BN_2"))
		model.add(MaxPooling2D(pool_size=(2, 2), name="block_2--POOL_max"))
		model.add(Dropout(0.25, name="block_2--DO"))

		# Third CONV block, (CONV => RELU)*2 => POOL.
		model.add(Conv2D(128, (3, 3), padding="same", name="block_3--CONV_1"))
		model.add(Activation("relu", name="block_3--ACT_relu_1"))
		model.add(BatchNormalization(axis=chanDim, name="block_3--BN_1"))
		model.add(Conv2D(128, (3, 3), padding="same", name="block_3--CONV_2"))
		model.add(Activation("relu", name="block_3--ACT_relu_2"))
		model.add(BatchNormalization(axis=chanDim, name="block_3--BN_2"))
		model.add(MaxPooling2D(pool_size=(2, 2), name="block_3--POOL_max"))
		model.add(Dropout(0.25, name="block_3--DO"))

		# Classify block, FC = > RELU => OUTPUT.
		model.add(Flatten())
		model.add(Dense(1024, name="block_end--FC_1"))
		model.add(Activation("relu", name="block_end--ACT_relu"))
		model.add(BatchNormalization(name="block_end--BN"))
		model.add(Dropout(0.5, name="block_end--DO"))
		# Output, use a 'softmax' ACT -- for single-label classification;
		#      or use a 'sigmoid' ACT -- for multi-label classification.
		model.add(Dense(classes, name="block_end--FC_2"))
		model.add(Activation(finalAct, name="block_end--ACT_output"))

		# Return the constructed network architecture.
		return model