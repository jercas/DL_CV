"""
Created on Sun May 27 17:26:00 2018

@author: jercas
"""

from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetReader:
	def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
		"""

		Parameters:
			dbPath: The path to our HDF5 dataset that stores our images and corresponding class labels.
			batchSize: The size of mini-batches to yield when training our network.
			preprocessors: The list of image preprocessors we are going to apply (i.e., MeanPreprocessor, ImageToArrayPreprocessor, etc.).
			aug: Defaulting to None, we could also supply a Keras ImageDataGenerator to apply data augmentation directly
				inside our HDF5DatasetGenerator.
			binarize: Typically we will store class labels as single integers inside our HDF5 dataset; however, as we
					know, if we are applying categorical cross-entropy or binary cross-entropy as our loss function, we
					first need to binarize the labels as one-hot encoded vectors – this switch indicates whether or not
					this binarization needs to take place (which defaults to True).
			classes: The number of unique class labels in our dataset. This value is required to accurately construct
					our one-hot encoded vectors during the binarization phase.
		"""
		# Store the batch size, preprocessors and data generator, whether of not the labels should be binarized, along
		#with the total number of classes.
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes

		# Open the HDF5 dataset for reading and determine the total number of entries in the dataset.
		self.db = h5py.File(dbPath)
		# Convenience variable used to access the total number of data points in the dataset.
		self.numImages = self.db["labels"].shape[0]


	def generator(self, passes=np.inf):
		# Initialize the epoch count.
		epochs = 0

		# This loop will run indefinitely until either:
		#1. Keras reaches training termination criteria.
		#2. We explicitly stop the training process (i.e., ctrl + c).
		while epochs < passes:
			# Loop over the HDF5 dataset over each batch of data points in the dataset.
			for i in np.arange(0, self.numImages, self.batchSize):
				# Extract the images and labels form the HDF5 dataset.
				images = self.db["images"][i : i+self.batchSize]
				labels = self.db["labels"][i : i+self.batchSize]
				# Check to see if the labels should be binarized.
				if self.binarize:
					labels = np_utils.to_categorical(labels, num_classes=self.classes)
				if self.preprocessors is not None:
					# Initialize the list of processed images.
					processedImages = []
					# Loop over the images.
					for image in images:
						# Loop over the preprocessors and apply preprocessor each to the each input image.
						for preprocess in self.preprocessors:
							image = preprocess.preprocess(image)
						# Update the list of processed image.
						processedImages.append(image)
					# Update the images array to be the processed images.
					images = np.array(processedImages)
				# If the data augmenator exists, apply it.
				if self.aug is not None:
					# Yielding a 2-tuple of the batch of images and labels to the calling Keras generator.
					(images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))
				# yield a tuple of images and labels
				yield (images, labels)

			# Increment the total number of epochs.
			epochs += 1


	def close(self):
		# Close the HDF5 dataset.
		self.db.close()