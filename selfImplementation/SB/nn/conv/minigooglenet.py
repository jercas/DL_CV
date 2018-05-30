"""
Created on Wed May 30 14:52:00 2018

@author: jercas
"""

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
# takes a set of inputs and concatenates them along a given axis, which in this case will be the channel dimension.
from keras.layers.merge import concatenate
from keras.layers import Input
# Using Model rather than Sequential allows to create a network graph with splits and forks like in the Inception module.
from keras.models import Model
from keras import backend as K

class MiniGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		"""
		Define the CONV=>ACT(ReLU)=>BN pattern/module.
		Parameters:
			x: The input layer to the function.
			K: The number of the given filters.
			kX: The horizontal size of the filter.
			kY: The vertical size of the filter.
			stride: The stride 2-tuple of the filter's movement.
			chanDim: The position of the channels dimension which is derived from either “channels last” or “channels
first” ordering.
			padding: The padding type/strategy of the convolution operation.
		Return:
			x: the constructed block.
		"""

		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)

		# Return the block.
		return x


	@staticmethod
	def inception_module(x, K_1x1, K_3x3, chanDim):
		"""
		Defining two CONV modules, then concatenate across the channel dimension
		Parameters:
			x: The input layer to the function.
			K_1x1: The number of the 1x1 *local features* filters.
			K_3x3: The number of the 3x3 *abstract features* filters.
			chanDim: The position of the channels dimension which is derived from either “channels last” or “channels
first” ordering.
		Return:
			x: the constructed block.
		"""

		# Branch that create two different size filters which would performed in parallel.
		conv_1x1 = MiniGoogLeNet.conv_module(x, K_1x1, 1, 1, (1, 1), chanDim, padding="same")
		conv_3x3 = MiniGoogLeNet.conv_module(x, K_3x3, 3, 3, (1, 1), chanDim, padding="same")
		# Concatenate the result across the channel dimension.
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		# Return the block.
		return x


	@staticmethod
	def downsample_module(x, K, chanDim):
		"""
		Defining the CONV module and POOL both responsible for reducing the spatial dimensions of input volume.
		Parameter:
			x: The input layer to the function.
			K: The number of the given filters.
			chanDim: The position of the channels dimension which is derived from either “channels last” or “channels
first” ordering.
		Return:
			x: the constructed block.
		"""

		# Branch that define the CONV module and POOL, respectively.
		# Valid padding to ensure the same output volume size as the output of the max-pooling layer.
		conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
		pool = MaxPooling2D((3,3), strides=(2, 2))(x)
		# Concatenate the result across the channel dimension.
		x = concatenate([conv_3x3, pool], axis=chanDim)

		# Return the block.
		return x


	@staticmethod
	def build(width, height, depth, classes):
		# Initialize the input shape to be "channels last" and the channels dimension itself.
		inputShape = (height, width, depth)
		chanDim = -1

		# If channels order is "channels first", modify the input shape and channels dimension.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Defining the model input and the first CONV module.
		input = Input(shape=inputShape)
		x = MiniGoogLeNet.conv_module(input, 96, 3, 3, (1,1), chanDim, padding="same")

		# Two Inception modules followed by a Downsample module.
		x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

		# Four Inception modules followed by a Downsample module.
		x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

		# Two Inception modules followed by a global-average pooling layer.
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		# Finally, end by a FC layer set.
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		# Create the defined model.
		model = Model(input, x, name="minigooglenet")

		# Visualizing the architecture of the builded model.
		model.summary()

		# return the constructed network architecture
		return model