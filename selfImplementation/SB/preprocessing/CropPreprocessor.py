"""
Created on Sun May 27 16:46:00 2018

@author: jercas
"""

import numpy as np
import cv2

class CropPreprocessor:
	def __init__(self, width, height, horizontal=True, inter=cv2.INTER_AREA):
		# Store the target image width, height of each cropped region whether or not horizontal flips should be included,
		#along with the interpolation method OpenCV will use for resizing.
		self.width  = width
		self.height = height
		self.horizontal = horizontal
		self.inter = inter


	def preprocess(self, image):
		# Initialize the list of crops.
		crops = []

		# Grab the width and height of the image then use these dimensions to define the corners of the image based.
		# (x;y)-coordinates of the four corners (top-left, top-right, bottom-right,bottom-left, respectively)
		(h, w) = image.shape[:2]
		coordinates = [
			[0,                 0,                  self.width,     self.height],
			[w - self.width,    0 ,                 w,              self.height],
			[w - self.width,    h - self.height,    w,              h],
			[0,                 h - self.height,    self.width,     h]
				]

		# Compute the center crop of the image.
		dW = int((w - self.width ) / 2.0)
		dH = int((h - self.height) / 2.0)
		coordinates.append([dW, dH, w - dW, h - dH])

		# Loop over each of the coordinates of the rectangular crops, extract each of the crops and resize each of them
		#to a fixed input size.
		for (startX, startY, endX, endY) in coordinates:
			crop = image[startY:endY, startX:endX]
			crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
			crops.append(crop)

		# Check to see if the horizontal flips should be taken.
		if self.horizontal:
			# Compute the horizontal mirrors/flips for each crop.
			mirrors = [cv2.flip(crop, 1) for crop in crops]
			crops.append(mirrors)
		# Return the 10-crops.
		return np.array(crops)