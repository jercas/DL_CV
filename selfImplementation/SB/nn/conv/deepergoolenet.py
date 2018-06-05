"""
Created on Mon Jun 4 15:49:00 2018

@author: jercas
"""

import tiny_imagenet_config as config
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import Input, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import keras.backend as K

class DeeperGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same", reg=0.0005, name=None):
		"""
		The convolution module of the GoogLeNet.

		Parameter:
			x: The input to the network.
			K: The number of filters the convolutional layer will learn.
			kX and kY: The filter size for the convolutional layer.
			stride: The stride (in pixels) for the convolution. Typically we’ll use a 11 stride, but we
				could use a larger stride if we wished to reduce the output volume size.
			chanDim: This value controls the dimension (i.e., axis) of the image channel. It is automatically set later in
				this class based on whether we are using “channels_last” or “channels_first” ordering.
			padding: Here we can control the padding of the convolution layer.
			reg: The L2 weight decay strength.
			name: Since this network is deeper than all others we have worked with in this book, we may wish to name the
				blocks of layers to help us (1) debug the network and (2) share/explain the network to others.
		Return:
			 x: Constructed module/layers block.
		"""
		# Initialize the CONV, BN and ReLU layer names.
		(convName, bnName, actName) = (None, None, None)
		# IF a module name wae supplied, prepend it.
		if name is not None:
			convName = name + "_conv"
			bnName   = name + "_bn"
			actName  = name + "_act"

		# Define a CONV=>BN=>ReLU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding, kernel_regularizer=l2(reg), name=convName)(x)
		x = Activation("relu", name=actName)(x)
		x = BatchNormalization(axis=chanDim, name=bnName)(x)

		# Return the block.
		return x


	@staticmethod
	def inception_module(x, K_1x1, K_3x3_reduce, K_3x3, K_5x5_reduce, K_5x5, K_1x1_Proj, chanDim, reg=0.0005, stage=""):
		"""
		The first branch is responsible for learning local 1x1 features.
		The second branch performs dimensionality reduction via a 1x1 convolution, followed by learning a larger filter
	size of 3x3.
		The third branch behaves similarly to the second branch, only learning 5x5 filters rather than 3x3 filters.
		Finally, the fourth branch applies max pooling.

		Parameters:
			x: The input to the network.
			K_1x1: The number of filters the first branch's convolution module will learn.
			K_3x3_reduce: The number of filters the second branch's first 1x1 convolution module will learn.
			K_3x3: The number of filters the second branch's second 3x3 convolution module will learn.
			K_5x5_reduce: The number of filters the third branch's first 1x1 convolution module will learn.
			K_5x5: The number of filters the third branch's second 5x5 convolution module will learn.
			K_1x1_Proj: The number of filters the fourth branch's 1x1 convolution module will learn.
			chanDim: This value controls the dimension (i.e., axis) of the image channel. It is automatically set later in
				this class based on whether we are using “channels_last” or “channels_first” ordering.
			reg: The L2 weight decay strength.
			stage: The prefix of the name of each convolution module/layer of the inception module.
		Return:
			x: Constructed module/layers block.
		"""
		# Define the first branch of the Inception module which consists of just 1x1 convolutions.
		first = DeeperGoogLeNet.conv_module(x, K_1x1, 1, 1, (1, 1), chanDim, reg=reg, name=stage + "_first")

		# Define the second branch of the Inception module which consists of 1x1 and 3x3 convolutions.
		second = DeeperGoogLeNet.conv_module(x,      K_3x3_reduce, 1, 1, (1, 1), chanDim, reg=reg, name=stage + "_second_1")
		second = DeeperGoogLeNet.conv_module(second, K_3x3,        3, 3, (1, 1), chanDim, reg=reg, name=stage + "_second_2")

		# Define the third branch of the Inception module which consists of 1x1 and 5x5 convolutions.
		third = DeeperGoogLeNet.conv_module(x,     K_5x5_reduce, 1, 1, (1, 1), chanDim, reg=reg, name=stage + "_third_1")
		third = DeeperGoogLeNet.conv_module(third, K_5x5,        5, 5, (1, 1), chanDim, reg=reg, name=stage + "_third_2")

		# Define the fourth branch of the Inception module which is the POOL projection.
		fourth = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=stage + "_pool")(x)
		fourth = DeeperGoogLeNet.conv_module(fourth, K_1x1_Proj, 1, 1, (1, 1), chanDim, reg=reg, name=stage + "_fourth")

		# Concatenate across the channel dimension.
		x = concatenate([first, second, third, fourth], axis=chanDim, name=stage + "_mixed")

		# Return the block.
		return x


	@staticmethod
	def build(width, height, depth, classes, reg=0.0005):
		# Initialize the input shape to be "channels last" and the channels dimension itself.
		inputShape = (height, width, depth)
		chanDim = -1

		# If channels order is "channels first", modify the input shape and channels dimension.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Define the first block, a sequence of CONV=>POOL=>(CONV*2)=>POOL layers.
		input = Input(shape=inputShape, name="Input")
		x = DeeperGoogLeNet.conv_module(input, 64,  5, 5, stride=(1, 1), chanDim=chanDim, reg=reg, name="block1")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
		x = DeeperGoogLeNet.conv_module(x, 64,  1, 1, stride=(1, 1), chanDim=chanDim, reg=reg, name="block2")
		x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, stride=(1, 1), chanDim=chanDim, reg=reg, name="block3")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

		# Define the second block, two inception module(3a, 3b) followed by a POOL operation.
		x = DeeperGoogLeNet.inception_module(x,  64,  96, 128, 16, 32, 32, chanDim=chanDim, reg=reg, stage="3a")
		x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim=chanDim, reg=reg, stage="3b")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

		# Apply five inception module(4a, 4b, 4c, 4d, 4e) on top of each other, then followed by a POOL layer.
		x = DeeperGoogLeNet.inception_module(x, 192,  96, 208, 16,  48,  64, chanDim=chanDim, reg=reg, stage="4a")
		x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24,  64,  64, chanDim=chanDim, reg=reg, stage="4b")
		x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24,  64,  64, chanDim=chanDim, reg=reg, stage="4c")
		x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32,  64,  64, chanDim=chanDim, reg=reg, stage="4d")
		x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim=chanDim, reg=reg, stage="4e")
		x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

		# Apply a POOL layer (global average) followed by a dropout layer to avoid the usage of computationally expensive
		#FC layers (not to mention, dramatically increased network size).
		x = AveragePooling2D((4, 4), name="pool5")(x)
		x = Dropout(0.5, name="do")(x)

		# Softmax classifier.
		x = Flatten(name="flatten")(x)
		x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
		x = Activation("softmax", name="softmax")(x)

		# Assemble the whole model.
		model = Model(input, x, name="googlenet")

		model.summary()
		plot_model(model, to_file='{0}/googlenet_visualization.png'.format(config.OUTPUT_DIR), show_shapes=True, show_layer_names=True)
		# Return the constructed network architecture.
		return model