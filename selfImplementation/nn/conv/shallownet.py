"""
Created on Thu Apr 19 11:09:00 2018

@author: jercas
"""
from keras.models import  Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
# FLATTEN: takes multi-dimensional volume and "flattens" it into a 1D array prior to feeding the inputs into the DENSE layers.
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
	# Define all network architectures inside a class to keep the code organized.
	@staticmethod
	def build(width, height, depth, classes):
		"""
			Parameters:
				width: The width of the input images(i.e., the number of columns in the matrix).
				height: The height of the input images(i.e., the number of rows in the matrix).
				depth: The number of channels in the input image.
				classes: The total number of classes/categories that out network should learn to predict.
			Return:
				model: the constructed network architecture.
		"""
		# Initialize the model along with the default input shape to be "channels last".
		model = Sequential()
		inputShape = (height, width, depth)

		# If we are using "channels first", update the input shape.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# Define the network architecture.
		# The fitst and only CONV layer -- contains 32 FILTERS(K) each of which are 3*3(i.e., square F*F filters).
		#                                  apply SAME PADDING to ensure the size of output of the convolution operation
		#                                  matches the input size.
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
		# ACT layer after CONV layer with the ReLU activation function.
		model.add(Activation("relu"))
		# Flattening multi-dimensional volume representation into a 1D array prior to feeding the inputs into the FC layer.
		model.add(Flatten())
		# FC layer, the number of output neurons = categories/classes
		model.add(Dense(classes))
		# Softmax activation for multiple classify.
		model.add(Activation("softmax"))

		return model