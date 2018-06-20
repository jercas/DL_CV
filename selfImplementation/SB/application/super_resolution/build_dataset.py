"""
Created on Sun Jun 17 17:08:00 2018

@author: jercas
"""

from SB.io.hdf5DatasetWriter import HDF5DatasetWriter
import super_resolution.config.super_resolution_config as config
from imutils import paths
from scipy import misc
import shutil
import random
import cv2
import os

# If the output directories do not exist, create them.
for path in [config.IMAGES, config.LABELS]:
	if not os.path.exists(path):
		os.mkdir(path)

# Grab the image paths and initialize the total number of crop processed.
print("[INFO] creating temporary images...")
imagePaths = list(paths.list_images(config.INPUT_IMAGES_PATH))
random.shuffle(imagePaths)
total = 0
"""
# Loop over the image paths.
for imagePath in imagePaths:
	# Load the input image.
	image = cv2.imread(imagePath)

	# Grab the dimensions of the input image and crop the image such that it tiles nicely when we generate the training
	#data + labels.
	# If did not take this step, the stride size would not fit and should crop patches outside of the image’s spatial dimensions.
	(h, w) = image.shape[:2]
	w -= int(w % config.SCALE)
	h -= int(h % config.SCALE)
	image = image[0:h, 0:w]

	# Generate training images we first need to down-scale the image by the scale factor... and then up-scale it back to
	#the original size -- this process allows us to generate low resolution inputs that we'll then learn to reconstruct
	#the high resolution version from.
	scaled = misc.imresize(image, 1.0 / config.SCALE, interp="bicubic")
	scaled = misc.imresize(scaled, config.SCALE / 1.0, interp="bicubic")

	# Generate the sub-windows for both inputs and targets.
	for y in range(0, h - config.INPUT_DIM + 1, config.STRIDES):
		for x in range(0, w - config.INPUT_DIM + 1, config.STRIDES):
			# Crop output the 'INPUT_DIM x INPUT_DIM' ROI from scaled image -- this ROI will serve as the input to network.
			crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]
			# Crop out the 'LABEL_SIZE x LABEL_SIZE' ROI from original image -- this ROI will be the target output from network.
			target = image[y + config.PAD:y + config.PAD + config.LABEL_SIZE, x + config.PAD:x + config.PAD + config.LABEL_SIZE]

			# Construct the crop and target output image paths.
			cropPath   = os.path.sep.join([config.IMAGES, "{}.png".format(total)])
			targetPath = os.path.sep.join([config.LABELS, "{}.png".format(total)])

			# Write the image to disk.
			cv2.imwrite(cropPath, crop)
			cv2.imwrite(targetPath, target)

			# Increment the crop total.
			total += 1
"""
# Grab the paths to the images.
print("[INFO] building HDF5 dataset...")
inputPaths  = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

# Initialize the HDF5 datasets.
inputWriter  = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM, config.INPUT_DIM, 3), config.INPUTS_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths), config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUTS_DB)

# Loop over images.
for (inputPath, outputPath) in zip(inputPaths, outputPaths):
	# Load the two images and add them to their respective datasets.
	inputImage = cv2.imread(inputPath)
	outputImage = cv2.imread(outputPath)
	# Because we training for filters not for accuracy, so we don't care about the label of image(hence why just specify a value of -1).
	# The “class label” is technically the output sub-window that we would try to train our SRCNN to reconstruct.
	inputWriter.add([inputImage], [-1])
	outputWriter.add([outputImage], [-1])

print("[INFO] clean up...")
# Close the HDF5 datasets.
inputWriter.close()
outputWriter.close()
# Delete the temporary output dictionaries.
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)