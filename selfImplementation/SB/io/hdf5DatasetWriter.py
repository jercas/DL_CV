"""
Created on Tue May 8 16:11:00 2018

@author: jercas
"""
import numpy as np
import h5py
import os

class HDF5DatasetWriter:
	def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
		"""
			Parameters:
				dims: controls the dimension or shape of the data which will be storing in the dataset. Just as .shape of a NumPy array.
					(if store the 'flattened' raw pixel intensities of the 28*28=784 MNIST dataset, then dims=(70000, 784)
					as there as 70,000 examples in MNIST, each with a dimensionality of 784)

					(if store the raw CIFAR-10 dataset, then dims=(60000, 32, 32, 3) as there as 60,000 total images in the
					CIFAR-10 dataset, each represented by a 32*32*3 RGB image)

					(if used to store VGG16's extracted feature, then dims=(N, 25088) as the output of the final POOL layer is 512*7*7
					and after flattened yields a feature vector of length 25,088. ps: N is the total number of images in the dataset)
				outputPath: where output HDF5 file stored on disk.
				dataKey(optional): the name of the dataset that will store the data which algorithm/NN will learn from, default value "images".
								(since in most case we'll be storing raw images in HDF5 format, however, when you want to storing features extracted
								from a CNN, it would be set as "features")
				bufSize: controls the size of in-memory buffer, which default value is 1000 indicate to 1,000 feature vectors/images.
						(once reache bufSize, we should flush the buffer to the HDF5 dataset)
		"""
		# Check to see if the output path exists, and if so, raise an exception since we don't want to overwrite an existing database.
		if os.path.exists(outputPath):
			raise ValueError("The supplied 'outputPath' already exists and cannot be overwritten. Manually delete the file before continuing.",
			                 outputPath)
		# Open the HDF5 database for writing and create two datasets:
		# one to store the images/features and another to store the class labels.from
		self.db = h5py.File(outputPath, "w")
		# Create a dataset with the dataKey name and the supplied dims - this is where we will store raw images/extracted features.
		self.data = self.db.create_dataset(dataKey, dims, dtype="float")
		# Create a second dataset for storing class labels(integer) for each record in the dataset.
		self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

		# Store the buffer size, then initialize the buffer itself along with the index into the datasets.
		self.bufSize = bufSize
		self.buffer = {"data": [], "labels":[]}
		self.idx = 0


	def add(self, rows, labels):
		"""
			Add data to buffer.
			Parameters:
				rows: the rows that will be adding to the dataset.
				labels: the class labels along with each corresponding row.
		"""
		# Add the rows and labels to the buffer.
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)
		# Check to see if the buffer needs to be flushed to disk.
		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()


	def flush(self):
		"""
			Write the buffers to disk then reset the buffer.
		"""
		# Store data in disk from the buffer.
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		# Flush buffer.
		self.buffer = {"data": [], "labels":[]}


	def storeClassLabels(self, classLabels):
		"""
			Create a dataset to store the actual class label names then store the class labels in a separate dataset.
		"""
		dt = h5py.special_dtype(vlen=str)
		labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
		labelSet[:] = classLabels


	def close(self):
		"""
			Check to see if there are any other entries left in the buffer that need to be flushed to disk as we will close the dataset.
		"""
		# If there are any left data, flush it to the disk.
		if len(self.buffer["data"]) > 0:
			self.flush()
		# close the dataset
		self.db.close()