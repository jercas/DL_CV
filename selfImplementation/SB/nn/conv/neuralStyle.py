"""
Created on Tue Jun 12 14:05:00 2018

@author: jercas
"""

from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
# An implementation of the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm used to iteratively minimize
#unconstrained nonlinear optimization problems.
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import cv2
import os

class NeuralStyle:
	def __init__(self, settings):
		# Store the settings dictionary.
		self.S = settings

		# Grab the dimensions of the input image.
		(w, h) = load_img(self.S["input_path"]).size
		self.dims = (h, w)

		# Load content image and style image, forcing the dimensions of our input image.
		self.content = self.preprocess(self.S["input_path"])
		self.style   = self.preprocess(self.S["style_path"])
		# Instantiate Keras variables from the input images.
		self.content = K.variable(self.content)
		self.style   = K.variable(self.style)

		# Allocate memory of output image, then combine the content, style, and output into a single tensor so they can
		#be fed through the network.
		self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
		# The goal will be to minimize our style loss, content loss, and total-variation loss based on this input tensor.
		self.input  = K.concatenate([self.content, self.style, self.output], axis=0)

		# Load the pre-trained model.
		print("[INFO] loading model...")
		self.model = self.S["net"](weights="imagenet", include_top=False, input_tensor=self.input)

		# Build a dictionary that maps the *name* of each layer inside the network to the actual layer *output*.
		layerMap = {layer.name: layer.output for layer in self.model.layers}

		# Extract features from the content layer, then extract the activations from the style image (index 0) and the
		#output image (index 2) -- these will serve as the style features, and output features from the *content* layer.
		contentFeatures = layerMap[self.S["content_layer"]]
		styleFeatures   = contentFeatures[0, :, :, :]
		outputFeatures  = contentFeatures[2, :, :, :]

		# Compute the feature reconstruction loss, weighting it appropriately.
		contentLoss = self.featureReconLoss(styleFeatures, outputFeatures)
		contentLoss *= self.S["content_weight"]

		# Initialize style loss along with the value used to weight each style layer (in proportion to the total number
		#of style layers.
		styleLoss = K.variable(0.0)
		styleWeight = 1.0 / len(self.S["style_layers"])

		# Loop over the style layers.
		for layer in self.S["style_layers"]:
			# Grab the current style layer and use it to extract the style features and output features from the *style layer*.
			styleOutput    = layerMap[layer]
			styleFeatures  = styleOutput[1, :, :, :]
			outputFeatures = styleOutput[2, :, :, :]

			# Compute the style reconstruction loss.
			T = self.styleReconLoss(styleFeatures, outputFeatures)
			styleLoss += (styleWeight * T)

		# Finish computing the style loss, compute the total variational loss, and then compute the total loss that combine
		#all three.
		styleLoss *= self.S["style_weight"]
		tvLoss = self.S["tv_weight"] * self.tvLoss(self.output)
		totalLoss = contentLoss + styleLoss + tvLoss

		# Compute the gradients out of the output image with respect to loss.
		grads = K.gradients(totalLoss, self.output)
		outputs = [totalLoss]
		outputs += grads

		# The implemention of L-BFGS we will be using requires that loss and gradients be *two separate functions* so here
		#we create a Keras function that can compute both the loss and gradients together and then return each separately
		#using two different class methods.
		self.lossAndGrads = K.function([self.output], outputs)


	def preprocess(self, image):
		# Load the input image (while resizing it to the desired dimensions) and preprocess it.
		image = load_img(image, target_size=self.dims)
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# Return the proprecessed image.
		return image


	def deprocess(self, image):
		"""
		Takes the output of our neural stylet transfer algorithm and de-processes it so we can save the image to disk.
		"""
		# Reshape the image, then reverse the zero-centering by *adding* back in the mean values across the ImageNet training set.
		image = image.reshape((self.dims[0], self.dims[1], 3))
		image[:, :, 0] += 103.939
		image[:, :, 1] += 116.779
		image[:, :, 2] += 123.680

		# Clip any values falling outside the range [0, 255] and convert the image to an unsigned 8-bit integer.
		image = np.clip(image, 0, 255).astype("uint8")

		# Return the deprocessed image.
		return image


	def gramMat(self, X):
		"""
		Compute the Gram matrix.
		"""
		# The gram matrix is the dot product between the input vectors and their respective transpose.
		features = K.permute_dimensions(X, (2, 0, 1))
		features = K.batch_flatten(features)
		features = K.dot(features, K.transpose(features))

		# Return the gram matrix.
		return features


	def featureReconLoss(self, styleFeatures, outputFeatures):
		"""
		Compute the content loss.
		"""
		# The feature reconstruction loss is the squared error between the style features and output features.
		# The content-loss is the L2 norm (sum of squared differences) between the features of our input image and the
		#features of the target, output image.
		return K.sum(K.square(outputFeatures - styleFeatures))


	def styleReconLoss(self, styleFeatures, outputFeatures):
		"""
		Compute the style loss.
		"""
		# Compute the style reconstruction loss where A is the gram matrix for the style image and G is the gram matrix
		#for the generated image.
		A = self.gramMat(styleFeatures)
		G = self.gramMat(outputFeatures)

		# Compute the scaling factor of the style loss, then finish computing the style reconstruction loss.
		scale = 1.0 / float((2 * 3 * self.dims[0] * self.dims[1]) ** 2)
		# Computes the L2 norm (sum of squared differences) between the Gram matrices and scales appropriately.
		loss = scale * K.sum(K.square(G - A))

		# Return the style reconstruction loss.
		return loss


	def tvLoss(self, X):
		# The total variance loss encourages spatial smoothness in the output page -- here we avoid border pixels to avoid
		#artifacts.
		(h, w) = self.dims
		A = K.square(X[:, :h-1, :w-1, :] - X[:,   1:, :w-1, :])
		B = K.square(X[:, :h-1, :w-1, :] - X[:, :h-1,   1:, :])
		loss = K.sum(K.pow(A+B, 1.25))

		# Return the total-variance loss.
		return loss


	def transfer(self, maxEvals=20):
		# Generate a random noise image that will serve as a placeholder array(a randomly generated placeholder NumPy array
		#with the spatial dimensions of our input content image), slowly modified as we run L-BFGS to apply style transfer
		#to take on both the content and style of our respective input images.
		X = np.random.uniform(0, 255, (1, self.dims[0], self.dims[1], 3)) - 128

		# Start looping over the desired number of iterations.
		for i in range(0, self.S["iterations"]):
			# Run L-BFGS over the pixels in the generated image to minimize the neural style loss.
			print("[INFO] starting iteration {} of {}...".format(i + 1, self.S["iterations"]))
			(X, loss, _) = fmin_l_bfgs_b(self.loss, X.flatten(), fprime=self.grads, maxfun=maxEvals)
			print("[INFO] end of iteration {}, loss: {:.4e}".format(i + 1, loss))

			# Deprocess the generated image and write it to disk.
			image = self.deprocess(X.copy())
			path = os.path.sep.join([self.S["output_path"], "iter_{}.png".format(i)])
			# Save the de-processed image at each iteration.
			cv2.imwrite(path, image)


	def loss(self, X):
		# Extract the loss value.
		X = X.reshape((1, self.dims[0], self.dims[1], 3))
		lossValue = self.lossAndGrads([X])[0]
		# Return the loss.
		return lossValue


	def grads(self, X):
		# Compute the loss and gradients.
		X = X.reshape((1, self.dims[0], self.dims[1], 3))
		output = self.lossAndGrads([X])
		grads = output[1].flatten().astype("float64")
		# Return the gradients.
		return grads