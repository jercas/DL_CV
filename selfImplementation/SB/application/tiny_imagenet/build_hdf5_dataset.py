"""
Created on Mon Jun 3 10:37:00 2018

@author: jercas
"""

import tiny_imagenet_config as config
# Using the LabelEncoder to encode the WordNet IDs as single integers.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from SB.io.hdf5DatasetWriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# Grab the paths to the training images, then extract the training class labels and encode them.
trainPaths = list(paths.list_images(config.TRAIN_PATH))
trainLabels = [path.split(os.path.sep)[-3] for path in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# Performing the stratified sampling from the training set to construct a testing set.
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(trainPaths, trainLabels,
                                                                    test_size=config.NUM_TEST_IMAGES,
                                                                    stratify=trainLabels, random_state=42)

# Load the validation filename => class from the file and then use these mappings to build the validation paths and labels list.
mappings  = open(config.VAL_MAPPINGS).read().strip().split("\n")
# Extract the top two columns, the image filename and the WordNet ID, in each row.
mappings  = [nameToClass.split("\t")[:2] for nameToClass in mappings]
# Extract the path and class label of each validation example from the mapping file.
valPaths  = [os.path.sep.join([config.VAL_PATH, mapping[0]]) for mapping in mappings]
# Transform the WordNet ID string to a unique class label integer by looping over the WordNet IDs in each row and applying
#the label encoder.
valLabels = le.transform([mapping[1] for mapping in mappings])

# Construct a list pairing the training, validation, and testing image paths along with their corresponding labels and
#output HDF5 files.
dataset = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val",   valPaths,   valLabels,   config.VAL_HDF5),
	("test",  testPaths,  testLabels,  config.TEST_HDF5)]

# Initialize the lists of RGB channel average.
(R, G, B) = ([], [], [])

# Loop over the dataset tuple.
for (nameOfHDF5, paths, labels, outputPath) in dataset:
	# Create the HDF5 writer.
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

	# Initialize the progress bar.nameOfHDF5
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
	proBar  = progressbar.ProgressBar(maxval=len(paths), widgets=widgets)
	proBar.start()

	# Loop over each image path.
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# Load the image from disk.
		image = cv2.imread(path)

		# If we are building the training dataset, then compute the mean of each channel in the image, then update the
		#respective lists.
		if nameOfHDF5 == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# Add the image and its label to the HFD5 dataset.
		writer.add([image], [label])
		proBar.update()
	proBar.finish()
	writer.close()

# Construct a dictionary of averages, then serialize the means to a JSON file.
print("[INFO] serialize the RGB mean...")
Dictionary = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(Dictionary))
f.close()