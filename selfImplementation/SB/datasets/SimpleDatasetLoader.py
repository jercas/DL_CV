# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:02:45 2018

@author: jercas
"""

import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# Store the image preprocessor
		self.preprocessors = preprocessors

		# If the preprocessors are None, initialize them as an empty list
		if self.preprocessors is None:
			self.preprocessors = []


	def load(self, imagePaths, verbose=-1):
		# Initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# Load the image and extract the class label assuming that path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# Check to see if preprocessors are not None.
			if self.preprocessors is not None:
				# Loop over the preprocessors and apply each to the image in a given sequential order.
				for preprocessor in self.preprocessors:
					image = preprocessor.preprocess(image)

			# Treat processed image as a "feature vector" by updating the data list followed by the labels.
			# image size: 32*32 pixels
			data.append(image)
			labels.append(label)

			# Show an update every 'verbose' images.
			if verbose > 0 and i > 0 and (i+1) % verbose == 0:
				print("[INFO] processed {0}/{1}".format(i+1, len(imagePaths)))
		# Return a tuple of the data and labels.
		return (np.array(data), np.array(labels))