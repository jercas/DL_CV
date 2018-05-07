"""
Created on Thu Apr 19 10:15:00 2018

@author: jercas
"""
# Warp this function inside the class.
# The benefit of defining a class to handle the type of image preprocessing rather than simply calling img_to_array on
#every single image is that we can now "chain preprocessor together" as we load datasets from disk.
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# Store the image data format. Default to None which indicates that the setting inside keras.json should be used.
		self.dataFormat = dataFormat


	def preprocess(self, image):
		# Apply the keras utility function that correctly re-arranges the dimensions of the image.
		# Return a new NumPy array with the channels properly ordered.
		return img_to_array(image, data_format=self.dataFormat)