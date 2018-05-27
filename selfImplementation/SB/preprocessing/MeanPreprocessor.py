"""
Created on Sat May 26 15:26:00 2018

@author: jercas
"""

import cv2

class MeanPreprocessor:
	def __init__(self, rMean, gMean, bMean):
		# Store the Red, Green, Blue channel averages computed across the entire training set.
		self.rMean = rMean
		self.gMean = gMean
		self.bMean = bMean


	def preprocessor(self, image):
		# Split the image into its respective Red, Green, Blue channels.
		# OpenCV represents images in BGR order rather than RGB, hence why return tuple has the signature (B, G, R) rather than (R, G, B)
		(B, G, R) = cv2.split(image.astype("float32"))
		# Subtract the means for each channel.
		R -= self.rMean
		G -= self.gMean
		B -= self.bMean
		# Merge the channels back together and return the resulting image.
		return cv2.merge([B, G, R])
