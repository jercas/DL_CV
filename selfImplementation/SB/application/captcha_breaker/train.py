"""
Created on Thu May 2 16:04:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD

from nn.conv.lenet import LeNet
from utils.captchahelper import preprocess
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output plot and model")
args = vars(ap.parse_args())

# Initialize the data and labels.
data = []
labels = []
# Loop over the input images.
for imagePath in paths.list_images(args["dataset"]):
	# Load the image, pre-process it, and store it in the data list.
	image = cv2.imread(imagePath)
	# Convert image to gray-scale.
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = preprocess(image, 28,28)
	# Convert image to an array compatible with keras and its channel ordering.
	image = img_to_array(image)
	data.append(image)
	# Extract the class label from the image path and update the labels list.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Scale the raw pixel intensities to the range [0, 1].
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# Data set split as 4:1.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# Convert the labels from integers to vectors.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Initialize the model.
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01, decay=0.01/15, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model.
print("[INFO] training model...")
History = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=15, batch_size=32, verbose=1)

# Evaluate model.
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Save model.
print("[INFO] serializing model...")
model.save("{}/weight.hdf5".format(args["output"]))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), History.history["acc"], label="train_acc")
plt.plot(np.arange(0, 15), History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("{}/plot.png".format(args["output"]))
plt.show()