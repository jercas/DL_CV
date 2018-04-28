"""
Created on Sat Apr 28 14:38:00 2018

@author: jercas
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import img_to_array
from keras.utils import np_utils

from nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces.")
ap.add_argument("-o", "--output", required=True, help="path to output plot.")
ap.add_argument("-m", "--model", required=True, help="path to output model.")
args = vars(ap.parse_args())

# Initialize the list of data and labels.
data = []
labels = []
count = 1
# Loop over the input images.
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
	print("[INFO] loading and pre-process images -- no.{}".format(count))
	# Load the image, pre-process it and then store it in the data list.
	image = cv2.imread(imagePath)
	# Convert image to gray-scale.
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Resize image size from original 64*64 to 28*28 pixels.
	image = imutils.resize(image, width=28)
	# Convert image to an array compatible with keras and its channel ordering.
	image = img_to_array(image)
	# Update training set.
	data.append(image)

	# Extract the class label from the image path since the dir's name is its label and update the labels list.
	label = imagePath.split(os.path.sep)[-3]
	label = "smiling" if label == "positives" else "not_smiling"
	labels.append(label)
	count+=1

# Scale the raw pixel intensities to range [0,1].
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Convert the labels from integers to vectors.
le = LabelEncoder()
labels = np_utils.to_categorical(le.fit_transform(labels), 2)

# Account for skew in the labeled data.
# Classtotals is a list consist of the number of each category.
classTotals = labels.sum(axis=0)
# Scale these totals to obtain the 'classweight' used to handle the class imbalance.
# yielding the array: [1, 2.56], implied that network will treat every instance of 'positive' as 2.56 instances of 'negative'.
# And helps combat the class imbalance issue by amplifying the per-instance loss by a larger weight when seeing 'positive' examples.
classWeight = classTotals.max() / classTotals

# Partition the data into training and testing splits.
# Parameter-stratify: often used in imbalance dataset, ensure the same class ratio in training set and test set.
#                     stratify = y, use the class ratio in y.(e.g., Training set: A:B=4:1, Test set: A:B=4:1, similarly.)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Initialize the model.
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model.
print("[INFO] training model...")
History = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

# Evaluate the model.
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Save the model.
print("[INFO] serializing model...")
model.save("{}.hdf5".format(args["model"]))

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
plt.savefig("{}.png".format(args["output"]))
plt.show()