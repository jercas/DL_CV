"""
Created on Tue May 29 10:10:00 2018

@author: jercas
"""

from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from SB.io.hdf5DatasetWriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="./dataset/train/train", help="path to input dataset")
ap.add_argument("-o", "--output", default="./dataset/hdf5/features.hdf5", help="path to output HDF5 file")
ap.add_argument("-b", "--batch_size", type=int, default=16, help="batch-size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())

DATA_PATH = args["dataset"]
OUTPUT_PATH = args["output"]
INPUT_SHAPE = (224, 224)
BATCH_SIZE = args["batch_size"]
BUFFER_SIZE = args["buffer_size"]

# Grab the list of images that we'll be describing the randomly shuffle them to allow for easy training and testing splits
#via array slicing during training time.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(DATA_PATH))
random.shuffle(imagePaths)

# Extract the class labels from the image paths then encode the labels.
labels = [path.split(os.path.sep)[-1].split(".")[0] for path in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# Load the ResNet50 network.
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)

# Initialize the HDF5 dataset writer, then store the class label names in the dataset.
# The final average pooling layer of ResNet50 is 2048-d, hence why supply a value of 2048 as the dimensionality to HDF5datasetWriter.
dataset = HDF5DatasetWriter((len(imagePaths), 2048), OUTPUT_PATH, dataKey="features", bufSize=BUFFER_SIZE)
dataset.storeClassLabels(le.classes_)

# Initialize the progress bar.
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
probar  = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets)
probar.start()

# Loop over the images in batches.
for i in np.arange(0, len(imagePaths), BATCH_SIZE):
	# Extract the batch of images and labels, then initialize the list of actual images that will be passed through the
	#network for feature extraction.
	batchPaths  = imagePaths[i: i+BATCH_SIZE]
	batchLabels = labels[i: i+BATCH_SIZE]
	batchImages = []

	# Followed by each pre-processor operate on each image.
	# Loop over the images and labels in the current batch.
	for (j, imagePath) in enumerate(batchPaths):
		# Load the input image and ensure the image is resized to feeding into the network.
		image = load_img(imagePath, target_size=INPUT_SHAPE)
		image = img_to_array(image)

		# Preprocess the input image by expanding the dimensions and subtracting the mean RGB pixels intensities from the
		#ImageNet dataset.
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		# Add the image to the batch.
		batchImages.append(image)

	# Pass the images through the network and extract the outputs as actual features.
	# Transform the matrix to a vector.
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=BATCH_SIZE)

	# Reshape the features so that each image is represented by a flattened feature vector of the 'MaxPooling2D' outputs.
	features = features.reshape((features.shape[0], 2048))
	# Add the features and labels to HDF5 dataset.
	dataset.add(features, batchLabels)
	probar.update(i)

# Close the HDF5 dataset.
dataset.close()
probar.finish()