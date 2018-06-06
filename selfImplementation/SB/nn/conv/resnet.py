"""
Created on Wed Jun 6 14:05:00 2018

@author: jercas
"""

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import keras.backend as K

class ResNet:
	@staticmethod
	def residual_module(x, K, stride, chanDim, red=False, reg=1e-4, bnEps=2e-5, bnMom=0.9):
		"""
		The residual module of the ResNet architecture.
		Parameters:
			x: The input to the residual module.
			K: The number of the filters that will be learned by the final CONV in the bottlenecks.
			stride: Controls the stride of the convolution, help reduce the spatial dimensions of the volume *without*
				resorting to max-pooling.
			chanDim: Define the axis which will perform batch normalization.
			red: Cause not all residual module will be responsible for reducing the dimensions of spatial volums -- the
				red boolean will control whether reducing spatial dimensions (True) or not (False).
			reg: Controls the regularization strength to all CONV layers in the residual module.
			bnEps: Controls the ε responsible for avoiding 'division by zero' errors when normalizing inputs.
			bnMom: Controls the momentum for the moving average.
		Return:
			x: Return the output of the residual module.
		"""

		# The shortcut branch of the ResNet module should be initialize as the input(identity) data.
		shortcut = x

		# The first block of the ResNet module -- 1x1 CONVs.
		bn1   = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
		act1  = Activation("relu")(bn1)
		# Because the biases are in the BN layers that immediately follow the convolutions, so there is no need to introduce
		#a *second* bias term since we had changed the typical CONV block order, instead of using the *pre-activation* method.
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

		# The second block of the ResNet module -- 3x3 CONVs.
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

		# The third block of the ResNet module -- 1x1 CONVs.
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

		# If we would like to reduce the spatial size, apply a CONV layer to the shortcut.
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

		# Add together the shortcut (shortcut branch) and the final CONV (main branch).
		x = add([conv3, shortcut])

		# Return the addition as the output of the Residual module.
		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
		# Initialize the input shape to be "channels last" and the channels dimension itself.
		inputShape = (height, width, depth)
		chanDim = -1

		# If channels order is "channels first", modify the input shape and channels dimension.
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Set the input and apply BN layer.
		input = Input(shape=inputShape)
		# Use BN layer as the first layer, acts as an added level of normalization.
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

		# Check if trained on the CIFAR dataset.
		if dataset == "cifar":
			# Apply the first and single CONV layer.
			x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

		# Loop over the number of stages (block names).
		for i in range(0, len(stages)):
			# Initialize the stride, then apply a residual module used to reduce the spatial size of the input volume.

			# If this is the first entry in the stage, we’ll set the stride to (1, 1), indicating that no downsampling
			#should be performed. However, for every subsequent stage we’ll apply a residual module with a stride of (2, 2),
			#which will allow us to decrease the volume size.
			stride = (1, 1) if i == 0 else (2, 2)

			# Once we have stacked stages[i] residual modules on top of each other, our for loop brings us back up to here
			#where we decrease the spatial dimensions of the volume and repeat the process.
			x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

			# Loop over the number of layers in the stage.
			for j in range(0, stages[i] - 1):
				# Apply a residual module.
				x = ResNet.residual_module(x, filters[i + 1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)

		# After stacked all the residual modules on top of each other, we would move to the classifier stage.
		# Apply BN=>ACT=>POOL, in order to avoid using dense/FC layers we would instead apply Global Averager POOL to reduce
		#the volume size to 1x1xclasses.
		x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)

		# Softmax classifier.
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)

		# Construct the model.
		model = Model(input, x, name="ResNet")

		model.summary()
		plot_model(model, to_file='./output/resnet_visualization.png', show_shapes=True, show_layer_names=True)
		# Return the build network architecture.
		return model