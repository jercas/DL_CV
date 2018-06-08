"""
Created on Fri Jun 8 16:52:00 2018

@author: jercas
"""

from keras.applications import imagenet_utils
import imutils

def  sliding_window(image, step, ws):
	"""
	Sliding window on top of the image.
	Parameters:
		image: The image that be going to process and loop over.
		step: The stride of the slide window.
		ws: The width and height of the window that will be extracted from the input image.
	Return:
		(x, y , image[y: y  +ws[1], x:x+ws[0]]): Returns a tuple containing the (x;y)-coordinates of the sliding window
											along with the ROI itself.
	"""
	# Slide a window across the image (loop over the (x,y)-coordinates of the image, incrementing their respective x and
	#y counters by the provided step size).
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# Yield the current window.
			yield (x, y , image[y: y+ws[1], x:x+ws[0]])


def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	"""
	The image pyramid technique.
	Parameters:
		image: The image that be going to process and loop over.
		scale: The subsample scale power at each layer of pyramid, controls how the image is resized at each layer.
			Typically the scale is set between 1.1-1.5 in 0.1 or 0.05 increments.
		minSize: The stopping criterion, indicates the minimum required width and height of the layer.
	Return:
		image: Resized/Subsampled image.
	"""
	# Yield the original image.
	yield image
	# Keep looping over the image pyramid.
	while True:
		# Compute the dimensions of the next image in the pyramid.
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# If the resized image does not meet the supplied minimum size, then stop constructing the pyramid.
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# Yield the next image in the pyramid.
		yield image


def batch_processing(model, batchROIs, batchLocs, labels, minProb=0.5, top=10, dims=(224, 224)):
	"""
	Accept a set of batch ROIs and batch (x,y)-locations, make predications on them, then return the labels and associated
	probabilities for the most confident classes
	Parameters:
		model: The Keras model that we will be using for classification.
		batchROIs: A NumPy array containing the batch of ROIs we will be classifying. This array is constructed in the
				exact same manner as the previous chapter where we build batches of images.
		batchLocs: The (x;y)-coordinates (in terms of the original image) of each ROI in batchROIs.
		labels: A class labels dictionary that will be maintained during the entire classification process. The key to
			this dictionary is the label name, and the value is a list of tuples, including bounding box coordinates and
			associated class label probability.
		minProb: The minimum required probability for a classification to be considered a valid detection. We use this
				parameter to filter out weak detections.
		top: The number of top-K predictions to be returned by the network.
		dims: Spatial dimensions of the bounding box (should match spatial dimensions of the network).
	Return:
	"""
	# Pass batch ROIs through trained model and decode the predictions.
	preds = model.predict(batchROIs)
	preds = imagenet_utils.decode_predictions(preds, top=top)