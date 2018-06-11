"""
Created on Mon Jun 11 21:16:00 2018

@author: jercas
"""

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2

def preprocess(image):
	# Load the input image, convert it to a keras-compatible array, expand the dimensions so we can pass it through the
	#model as a batch that contain only one image, and then finally preprocess it for input to the pre-trained model.
	image = load_img(image)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)

	# Return the preprocessed image.
	return image


def deprocess(image):
	# As using the 'channels last' ordering so ensure the RGB channels are the last dimension in the matrix.
	image = image.reshape((image.shape[1], image.shape[2], 3))

	# 'Undo' the preprocessing done for Inception model to bring the image back into the range [0, 255].
	image /= 2.0
	image += 0.5
	image *= 255.0
	# Ensures any values that fall outside the range [0;255] are clipped.
	image = np.clip(image, 0, 255).astype("uint8")

	# Convert the processing order from 'RGB' to OpenCv assumes 'BGR' order.
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	# Return the deprocessed image.
	return image


def resize_image(image, size):
	# Resize the image.
	clone = np.copy(image)
	# The scale of both the width and height are determined by dividing the requested output size by the shape of the
	#image prior to resizing,
	resized = ndimage.zoom(clone, (1, float(size[0]) / clone.shape[1], float(size[1]) / clone.shape[2], 1), order=1)
	# Return the resized image.
	return resized


def eval_loss_and_gradients(X):
	# Fetch the loss and gradients given the input.
	output = fetchLossGrads([X])
	(loss, grad) = (output[0], output[1])
	return (loss, grad)


def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):
	"""
	Loop over a number of iterations, then compute the loss and gradients for input at each epoch, finally apply the actual
	gradient ascent step.
	Parameters:
		X: The our input tensor (i.e., the input image).
		iters: The total number of iterations to run for.
		alpha: The step size/learning rate when applying gradient descent.
		maxLoss: If the loss exceeds maxLoss, then terminate the gradient ascent process early, preventing from generating
			artifacts in our output image.
	Return:
		X: The output result of gradient ascent.
	"""
	# Loop over number of iterations.
	for i in range(0, iters):
		# Compute the loss and gradient.
		(loss, grad) = eval_loss_and_gradients(X)

		# If the loss is greater than the max loss, break from the loop early to prevent strange effects.
		if loss > maxLoss:
			break
		# Take a step.
		print("[INFO] Loss at {}: {}".format(i, loss))
		X += alpha * grad
	# Return the output of gradient ascent.
	return X



# Construct the parameter parser and parse the parameters.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output dreamed image")
args = vars(ap.parse_args())
# Define the dictionary that include (1) the layers are going to use for dream and (2) their respective weights (i.e.,
#the larger the weight, the more layer contributes to the dream make).
"""
	Lower layers can be used to generate edges and geometric patterns while high layers result in the injection of 
trippy visual patterns, including hallucinations of dogs, cats, and birds.
"""
LAYERS = {
	"mixed2": 2.0,
	"mixed3": 0.5,
}

# Define the number of octaves, octave scale, alpha (step for gradient ascent), number of iterations and max loss --
#tweaking these values will produce different dreams.
NUM_OCTAVES = 3         # The number of octaves (resolutions) be are going to generate.
OCTAVES_SCALE = 1.4     # Each successive octave, controlled by OCTAVE_SCALE will be 1.4x larger than the previous on (i.e., 40% larger).
ALPHA = 0.001           # The step size for gradient ascent.
NUM_ITER = 50           # The total number of gradient ascent operations.
MAX_LOSS = 10.0         # Early stopping criteria.

# Indicate that Keras *should not* be update the weights of any layer during the deep dream.
# This function call disables all training related operations.
K.set_learning_phase(0)

# Load the pre-trained Inception model from disk, then grab a reference variable to the input tensor of the model (
#which we'll then be using to perform the CNN hallucination).
print("[INFO] loading the Inception model...")
model = InceptionV3(weights="imagenet", include_top=True)
# Initialize the dream variable — this tensor holds our actual generated hallucination image.
dream = model.input
# Define loss value, then build a dictionary that maps the *name* of each layer insides of Inception to the actual
#*layer* object itself -- we'll need this mapping when building the loss of dream.
loss = K.variable(0.0)
layerMap = {layer.name: layer for layer in model.layers}

# Loop over the layers that will be utilizing in the dream.
for layerName in LAYERS:
	# Grab the output (i.e., the activations) of the layer we will use for dreaming, then add the L2-norm of the features
	#to the layer to the loss (we use array slicing here to avoid border artifacts caused by border pixels).
	x = layerMap[layerName].output
	# The corresponding weight/contribution of the layer to the output dream.
	coeff = LAYERS[layerName]
	scaling = K.prod(K.cast(K.shape(x), "float32"))
	loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream with respect to loss and then normalize.
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
outputs = [loss, grads]
fetchLossGrads = K.function([dream], outputs)

# Load and preprocess the input image, then grab the original input and width.
image = preprocess(args["image"])
# Extracts the spatial dimensions (height and width) from the .shape tuple (we’ll need these values when constructing the
#octaves and resizing the image).
dims = image.shape[1:3]

"""
	In order to perform deep dreaming, we need to build a list of dimensions we are going to resize our input image to. 
	Each resized image is called an octave, therefore the dimensions of each octave can be stored in a list named octaveDims.
"""
# In order to perform deep dreaming we need to build multiple scales of the input image (i.e., set of images at lower and
#lower resolutions) -- this list stores the spatial dimensions that we will be resizing input image to.
# Initialize octaveDims with the current dims of the input image。
octaveDims = [dims]

# Loop over the number of octaves (resolutions) be going to generate.
for i in range(1, NUM_OCTAVES):
	# Compute the spatial dimensions (i.e., width and height) for the current octave, then update the dimension list.
	size = [int(d / (OCTAVES_SCALE ** i)) for d in dims]
	octaveDims.append(size)

# Reverse the octave dimensions list so that the *smallest* dimensions are at the *front* of the list.
octaveDims = octaveDims[::-1]

# Clone the original image and then create a resized input image that matches the smallest dimensions.
original = np.copy(image)
shrunk = resize_image(image, octaveDims[0])

# Loop over the octave dimensions from smallest to largest.
for (o, size) in enumerate(octaveDims):
	# Resize the image and then apply gradient ascent.
	print("[INFO] starting octave {}...".format(o))
	image = resize_image(image, size)
	image = gradient_ascent(image, iters=NUM_ITER, alpha=ALPHA, maxLoss=MAX_LOSS)

	# To compute the loss detail we need two images: (1) the shrunk image that has been upscaled to the current octave and
	#(2) the original image that has been downscaled to the current octave.
	upscaled = resize_image(shrunk, size)
	downscaled = resize_image(original, size)

	# The lost detail is computed via a simple subtraction which immediately back in to the image applied gradient ascent to.
	lost = downscaled - upscaled
	image += lost

	# Make the original image be the new shrunk image so that the process can be repeated.
	shrunk = resize_image(original, size)

# Deprocess dream and save it to disk.
image = deprocess(image)
cv2.imwrite(args["output"], image)