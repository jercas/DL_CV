"""
Created on Wed Jun 13 16:27:00 2018

@author: jercas
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
"""
	 Transposed convolution layers, sometimes referred to as fractionally-strided convolution or (incorrectly) deconvolution,
	are used when we need a transform going in the opposite direction of a normal convolution.
	
	 The generator of our GAN will accept an input N-dimensional vector (i.e., a list of numbers, but a volume like an image) 
	and then transform this N-dimensional vector into an output image.
	
	 This process implies that we need to *reshape* and then *upscale* this vector into a volume as it passes through our 
	network to accomplish this reshaping and upscaling we’ll need transposed convolution.
	
	 We can thus look at transposed convolution as method to:
	1. Accept an input volume from a previous layer in the network;
	2. Produce an output volume that is larger than the input volume;
	3. Maintain a connectivity pattern between the input and output.
	
	 In essence, our transposed convolution layer will reconstruct our target spatial resolution and perform a normal 
	convolution operation, utilizing fancy zero-padding schemes to ensure our output spatial dimensions are met.
"""
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape

class DCGAN:
	@staticmethod
	def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):
		"""
		Build the generator of the GAN which intended to create synthetic images.
		Parameters:
			dim: The target spatial dimensions (width and height) of the generator after reshaping.
			depth: The target depth of the volume after reshaping.
			channels: The number of channels in the output volume from the generator (i.e., 1 for gray-scale images and 3 for RGB images).
			inputDim: Dimensionality of the randomly generated input vector to the generator.
			outputDim: Dimensionality of the output fully-connected layer from the randomly generated input vector.
		Return:
			model: The built model.
		"""
		# Initialize the model along with the input shape to be "channels last" and the channels dimension itself.
		model = Sequential()
		inputShape = (dim, dim, depth)
		chanDim = -1

		# First set of FC => RELU => BN layers.
		# FC layer will have an input dimension of inputDim (the randomly generated input vector) and then an output
		#dimensionality of outputDim. Typically, outputDim will be larger than inputDim.
		model.add(Dense(input_dim=inputDim, units=outputDim))
		model.add(Activation("relu"))
		model.add(BatchNormalization())

		# Second set of FC => RELU => BN layers, this time preparing the number of FC nodes to equal the number of units in inputShape.
		# Even though we are still utilizing a flattened representation, we need to ensure the output of this FC layer can be
		#reshaped to our target volume size (i.e., inputShape).
		model.add(Dense(dim * dim * depth))
		model.add(Activation("relu"))
		model.add(BatchNormalization())

		# Reshape the output of the previous layer set, upsample + apply a transposed convolution, ReLU and BN.
		# A call to Reshape while supplying inputShape allows us to create a 3D volume from the fully-connected layer.
		# WARNING: this reshaping is only possible due to the fact that the number of output nodes FC layer matches the target inputShape.
		model.add(Reshape(inputShape))
		model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		# Apply another upsample and transposed convolution but this time output the TANH activation.
		model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
		model.add(Activation("tanh"))

		# Review the constructed model.
		model.summary()
		# Return the build generator model.
		return model


	@staticmethod
	def build_discriminator(width, height, depth, alpha=0.2):
		"""
		Build the discriminator of the GAN used to classify whether an input is real of fake.
		Return:
			model: The built model.
		"""
		# Initialize the model along with the input shape to be "channels last" and the channels dimension itself.
		model = Sequential()
		inputShape = (width, height, depth)
		chanDim = -1

		# First set of CONV => RELU => BN layers.
		model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=inputShape))
		model.add(LeakyReLU(alpha=alpha))
		model.add(BatchNormalization(axis=chanDim))

		# Second set of CONV => RELU => BN layers.
		model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=inputShape))
		model.add(LeakyReLU(alpha=alpha))
		model.add(BatchNormalization(axis=chanDim))

		# First and the only set of FC => RELU layers.
		model.add(Flatten())
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=alpha))

		# Sigmoid layer outputting a single value.
		model.add(Dense(1))
		# Using a sigmoid here to capture the probability of whether the input image is real or synthetic.
		model.add(Activation("sigmoid"))

		# Review the constructed model.
		model.summary()
		# Return the build generator model.
		return model