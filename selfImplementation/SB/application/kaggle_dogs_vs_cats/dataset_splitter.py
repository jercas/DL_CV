"""
Created on Thu May 24 16:30:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import dogs_vs_cats_config as config
from SB.io.hdf5DatasetWriter import HDF5DatasetWriter
from SB.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor

from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# Grab the paths to the images.
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [path.split(os.path.sep)[-1].split(".")[0] for path in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
# Loop over each of the possible class labels and show them.
for (i, label) in enumerate(le.classes_):
	print("{}. {}.".format(i + 1, label))

# Perform stratified sampling from the training set to build the testing split from the training data.
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(trainPaths,
                                                                    trainLabels,
                                                                    test_size=config.NUM_TEST_CLASSES,
                                                                    stratify=trainLabels,
                                                                    random_state=42)
# Perform another stratified sampling, this time to build the validation data split from the remainder of training set.
(trainPaths, valPaths, trainLabels, valLabels) = train_test_split(trainPaths,
                                                                    trainLabels,
                                                                    test_size=config.NUM_VAL_CLASSES,
                                                                    stratify=trainLabels,
                                                                    random_state=42)

# Construct a list pairing the training, validation and testing image paths along with their corresponding labels and
#output HDF5 files, each element is a 4-tuple entry.
datasets = [("train", trainPaths, trainLabels, config.TRAIN_HDF5),
            ("val",   valPaths,   valLabels,   config.VAL_HDF5),
            ("test",  testPaths,  testLabels,  config.TEST_HDF5)]

# Initialize the image preprocessor used to resize the images (keep the aspect ratio of the image)
#and the lists used to store the averages pixel intensities for each channel.
aap = AspectAwarePreprocessor(width=256, height=256)
(R, G, B) = ([], [], [])

# Loop over the dataset tuple.
for (dtype, paths, labels, outputPath) in datasets:
	# Create the HDF5 writer.
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

	# Initialize the progress bar.
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ",progressbar.ETA()]
	proBar  = progressbar.ProgressBar(maxval=len(paths), widgets=widgets)
	proBar.start()

	# Loop over the image paths.
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# Load the image and process it.
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# If we are building the training dataset, then compute the mean of each channel in the image, then update the
		#repective list prepare for 'Mean Subtraction Normalization'.
		if dtype == "train":
			(b, g, r)= cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# Add the image and label into the HDF5 dataset.
		writer.add([image], [label])
		proBar.update(i)

	proBar.finish()
	writer.close()

# Construct a dictionary of averages, the serialize the means to a JSON file.
print("[INFO] serializing means...")
# Compute the mean value of each channel's average pixel intensity across all images in the training set.
mean = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(mean))
f.close()