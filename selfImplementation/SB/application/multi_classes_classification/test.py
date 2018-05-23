"""
Created on Wed May 23 14:30:00 2018

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
ap.add_argument("-m", "--model", type=str, default="./model/pokedex.hdf5", help="path to trained model model")
ap.add_argument("-l", "--labelbin",type=str, default="./model/lb.pickle", help="path to label binarizer")
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

# Load the trained convolutional neural network and the multi-classes binarizer.
print("[INFO] load model...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# Classify the input image then find the indexes of the class label with *largest probability*.
print("[INFO] classifying image...")
probability = model.predict(image)[0]

# Extract the highest probability of the class label.
idx = np.argmax(probability)
label = lb.classes_[idx]

# Mark our prediction as "correct" of the input image filename contains the predicted label text (obviously this makes the
#assumption that you have named your testing image files this way).
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# Prepare the class labels + associated confidence values for overlay on the output image.
label = "{}: {:.2f}% ({})".format(label, probability[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# Show the probabilities for each of the individual labels.
for (label, prob) in zip(lb.classes_, probability):
	print("{}: {:.2f}%".format(label, prob * 100))

# Show the output image.
cv2.imshow("Output", output)
cv2.waitKey(0)