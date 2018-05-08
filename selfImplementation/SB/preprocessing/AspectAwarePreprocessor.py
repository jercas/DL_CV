"""
Created on Mon Apr 7 15:13:00 2018

@author: jercas
"""

import imutils
import cv2

class AspectAwarePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# Store the target image width, height, and interpolation method used when resizing.
		self.width  = width
		self.height = height
		self.inter  = inter


	def preprocess(self, image):
		"""
			Determine the shortest dimension and resize along it.
			Then crop the image along the largest dimension to obtain the target width and height.

			Parameters:
				image: Input to be processed image.
			Return:
				image: Processed image.
		"""
		# Grab the dimensions of the image and then initialize the deltas to use when cropping.
		(h, w) = image.shape[:2]
		# Determine the delta offsets we'll be using when cropping along the larger dimension to maintain respect ratio.
		dW = 0
		dH = 0
		# If the width is smaller than the height, then resize along the width(i.e., the smaller dimension) and then
		#update the delta to crop the height to the desired dimension.
		if w < h:
			# Calculate the ratio of the width and construct the dimensions.
            #         r = width / float(w)
            # return  dim = (width, int(h * r))
			image = imutils.resize(image, width=self.width, inter=self.inter)
			dH = int((image.shape[0] - self.height) / 2.0)
		# Otherwise the height is smaller than the width, then resize along the height(i.e., the smaller dimension) and then
		#update the delta to crop the height to the desired dimension.
		else:
			# Calculate the ratio of the width and construct the dimensions.
			#         r = height / float(h)
			# return  dim = (int(w * r), height)
			image = imutils.resize(image, height=self.height, inter=self.inter)
			dW = int((image.shape[1] - self.width) / 2.0)

		# After resized, re-grab the width and height, followed by performing the crop.
		(h, w) =image.shape[:2]
		image = image[dH:h - dH, dW:w - dW]

		# Finally, resize the image(maintain aspect-ratio) to the provided spatial dimensions to ensure output image
		#is always a fixed size.
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
