"""
Created on Thu May 2 15:26:00 2018

@author: jercas
"""
import imutils
import cv2

def preprocess(image, width, height):
	"""
		Preprocessing image for train phase.
		Parameter:
			image: The inout image that we are going to pad and resize.
			width: The target output width of the image.
			height: The target output height of the image.
		Return:
			image: Preprocessed image.
	"""
	# Grab the dimensions of the image, then initialize the padding value.
	(h, w) = image.shape[:2]

	# If the width is greater than the height then resize along the width, otherwise, resize along the height.
	if w > h:
		image = imutils.resize(image, width=width)
	else:
		image = imutils.resize(image, height=height)

	# Determine the padding values for the width and height to obtain the target dimensions.
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# Pad the image then apply one more resizing to handle any rounding issues, ensure all images are the same width and height.
	image = cv2.copyMakeBorder(image, top=padH, bottom=padH, left=padW, right=padW, borderType=cv2.BORDER_REPLICATE)

	# Don't immediately call cv2.resize is because first need to consider the 'aspect ratio' of the input image and attempt
	#to pad it correctly first. If do not consider the image aspect ratio, then resized image will become distorted
	image = cv2.resize(image, (width, height))

	return image