"""
Created on Thu May 2 18:52:00 2018

@author: jercas
"""
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from selfImplementation.utils.captchahelper import preprocess

from imutils import paths
from imutils import contours

import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images.")
ap.add_argument("-m", "--model", required=True, help="path to the input model.")
args = vars(ap.parse_args())

# Load the pre-trained model.
print("[INFO] loading pre-trained model...")
model = load_model(args["model"])
# Randomly sample a few of the input images.
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# Loop over the to be predicted image paths.
for imagePath in imagePaths:
	# Load the image and convert it to gray-scale.
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Then pad the image to ensure digits caught only the border of the image are retained.
	gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
	# Threshold the image to reveal the digits.
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# Find contours in the image, keeping only the four largest ones, then sort them from left-to-right as a captcha shows.
	# cv2.findContours()--returns a list of (x.y)--coordinates that specify the outline of each individual digit.
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# First sort stage: sorts the contours by its size, keeping only the largest four outlines.
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
	# Since there is no guaranteed "spatial ordering imposed" on these contours and we always read digits from left-to-right.
	# So we need to sort the contours from left-to-right(accomplished via imutils.contours()) instead of just sorted as its size.
	cnts = contours.sort_contours(cnts)[0]

	# Initialize the output image as a "gray-scale" image with 3 channels along with the output predictions.
	output = cv2.merge([gray] * 3)
	predictions = []

	# Loop over the contours.
	for c in cnts:
		# Compute the bounding box for contour then extract the digit.
		(x, y, w, h) = cv2.boundingRect(c)
		# Extract the ROI.
		roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
		# Pre-process the ROI then classify it .
		roi = preprocess(roi, 28,28)
		roi = img_to_array(roi)
		# Input image is now represented as a NumPy array of shape(inputShape[0]-width, inputShape[1]-height, 3-depth).
		# Then expand the shape's dimension to (1-batch_size(when predict phase, it equals to 1),inputShape[0]-width, inputShape[1]-height, 3-depth).
		roi = np.expand_dims(roi, axis=0) / 255.0
		prediction = model.predict(roi).argmax(axis=1)[0]+1
		predictions.append(str(prediction))

		# Draw the prediction on the output image.
		cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
		cv2.putText(output, str(prediction), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

	# Show the output image.
	print("[INFO] captcha: {}".format("".join(predictions)))
	cv2.imshow("Output", output)
	cv2.waitKey()