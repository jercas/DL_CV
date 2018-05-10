"""
Created on Wed May 9 09:37:00 2018

@author: jercas
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
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
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=100, help="size of feature extraction buffer")
args = vars(ap.parse_args())

BATCH_SIZE = args["batch_size"]

# Grab the list of images that we'll be describing then randomly shuffle them to allow for easy training and testing splits
#via array slicing during training time.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# We shuffle the order of the dataset instead of shuffle them on the fit phase, because of we won't be able to perform this
	#shuffle in memory -- therefore, we shuffle the image paths before we extract the features.
random.shuffle(imagePaths)

# Extract the class labels from the image paths then encode the labels.
# Splitingthe path into an array based on the path separator('/' on Unix and '\' on Windows).
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
# Transform/Encode each "string" label to a single "integer" label, e.g., bluebell->1, buttercup->2(single integer! not a one-hot vector).
labels = le.fit_transform(labels)

# Load the VGG16 network.
print("[INFO] loading network...")
# include_top=False -- supplying this value indicates that the final FC layers should not be included in the architecture(we'll
#obtain the feature values after the final POOL layer rather than the probabilities produced by the softmax classifier in the FC layers).
model = VGG16(weights="imagenet", include_top=False)

# Initialize the HDF5 dataset writer, then store the class label names in the dataset.
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),     # The dimension of the dataset.
                            args["output"],
                            dataKey="features",
                            bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# Initialize the progress bar.
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
probar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# Loop over the images in patches.
for i in np.arange(0, len(imagePaths), BATCH_SIZE):
	# Extract the batch of images and labels, then initialize the list of actual images that will be passed through the
	#network for feature extraction.
	batchPaths  = imagePaths[i:i + BATCH_SIZE]
	batchLabels = labels[i:i + BATCH_SIZE]
	batchImages = []

	# Loop over the images and labels in the current batch.
	for (j, imagePath) in enumerate(batchPaths):
		# Load the input image using the Keras helper utility while ensuring the image is resized to 224*224 pixels.
		image = load_img(imagePath, target_size=(224, 224))
		# Convert to a Keras-compatible array.
		image = img_to_array(image)
		# Expanding the dimensions(1, 224, 224, 3) as convenience input for CNN.
		image = np.expand_dims(image, axis=0)
		# Subtracting the mean RGB pixel intensity from the ImageNet dataset.
		image = imagenet_utils.preprocess_input(image)
		# Add the image to the batch.
		batchImages.append(image)

	# Pass the images through the network and use the outputs as actual extracted features.
	# np.vstack -- Vertically stack images such that shape them as (N, 224, 224, 3) where N is the size of the batch.
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=BATCH_SIZE)
	# Reshape the features so that each image is represented by a "FLATTENED feature vector(N * 25088)" of the "MaxPooling2D" layer outputs.
	features = features.reshape((features.shape[0], 512 * 7 * 7))

	# Add the features and labels ti HDF5 dataset.
	dataset.add(features, batchLabels)
	probar.update(i)

# Close the input stream.
dataset.close()
# End the probar.
probar.finish()
