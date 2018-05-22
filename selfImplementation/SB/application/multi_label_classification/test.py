"""
Created on Tue May 22 16:26:00 2018

@author: jercas
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="./model/fashion.model", help="path to trained model model")
ap.add_argument("-l", "--labelbin",type=str, default="./model/mlb.pickle", help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Load the image.
image = cv2.imread(args["image"])
# Make a copy called output for display purposes.
output = image.copy()
# Pre-process the image for classification（preprocess the image in the same manner as we preprocessed our training data）.
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Load the trained convolutional neural network and the multi-label binarizer.
print("[INFO] load model...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# Classify the input image then find the indexes of the two class labels with *largest probability*.
print("[INFO] classifying image...")
a = model.predict(image)
probability = model.predict(image)[0]

# Extract the top two class labels indices by sorting the array indexes by their associated probability in descending order.
# Then grabbing the first two class label indices which are thus the top-2 predictions from our network.
# ********************************************************************************************************************
# ***Tips: You can modify this code to return more class labels if you wish. I would also suggest thresholding the ***
# ***probabilities and only returning labels with > N% confidence.                                                 ***
# ********************************************************************************************************************
idxs = np.argsort(probability)[::-1][:2]

# Prepare the class labels + associated confidence values for overlay on the output image.
# Loop over the indexes of the high confidence class labels.
for (i, j) in enumerate(idxs):
	# Build the label and draw the label on the image.
	label ="{}: {:.2f}%".format(mlb.classes_[j], probability[j] * 100)
	output = imutils.resize(output, width=400)
	cv2.putText(output, label, (10, (i*30) +25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the probabilities for each of the individual labels.
for (label, prob) in zip(mlb.classes_, probability):
	print("{}: {:.2f}%".format(label, prob * 100))

# Show the output image.
cv2.imshow("Output", output)
cv2.waitKey(0)