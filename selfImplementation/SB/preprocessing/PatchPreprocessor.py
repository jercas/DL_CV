"""
Created on Sun May 27 16:11:00 2018

@author: jercas
"""

from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
	def __init__(self, width, height):
		# Store the target width and height of the image.
		self.width  = width
		self.height = height


	def preprocess(self, image):
		# Extract a random crop from the image with the target width and height.
		# max_patches=1 -- indicating that we only need a single random patch from the input image.
		return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]