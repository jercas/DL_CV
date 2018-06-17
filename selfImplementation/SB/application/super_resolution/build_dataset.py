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

# Loop over the image paths.
for imagePath in imagePaths:
	# Load the input image.
	image = cv2.imread(imagePath)

	# Grab the dimensions of the input image and crop the image such that it tiles nicely when we generate the training
	#data + labels.
	(h, w) = image.shape[:2]
	w -= int(w % config.SCALE)
	h -= int(h % config.SCALE)
	image = image[0:h, 0:w]

