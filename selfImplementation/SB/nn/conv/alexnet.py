"""
Created on Sun May 27 21:35:00 2018

@author: jercas
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.regularizers import l2

class AlexNet:
	@staticmethod
	def build(width, height, depth, classes, reg=0.0002):
		"""
		Model builder.
		Parameter:
			width: The width dimension of image/number of horizontal pixels.
			height: The height dimension of image/number of vertical pixels.
			depth: The number of image channels.
			classes: The total number of class labels in the dataset.
			reg: Control the amount of L2 Regularization.
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

		# CONV-Block1 #1: first CONV=>RELU=>POOL layer set.
		model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))
		# CONV-Block2 #2: second CONV=>RELU=>POOL layer set.
		model.add(Conv2D(256, (5, 5), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))
		# CONV-Block3 #3:  (CONV=>RELU)*3=>POOL.
		model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))
		# FC-Block1 #4: first FC=>RELU layer set.
		model.add(Flatten())
		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.5))
		# FC-Block2 #5: second FC=>RELU layer set.
		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.5))
		# SOFTMAX-Block #6: softmax classifier.
		model.add(Dense(classes, kernel_regularizer=l2(reg)))
		model.add(Activation("softmax"))

		# Return the constructed network architecture.
		model.summary()
		return model