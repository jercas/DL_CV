"""
Created on Thu Apr 19 15:45:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing.SimplePreprocessor import SimplePreprocessor
from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# Grab the list of images that we'll be describing.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the image preprocessors.
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel grey intensities to the range [0, 1].
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Partition the data into training and testing(ignore validation) splits using 75% of the data for training and the
#remaining 25% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# One-Hot coding:Convert the labels from integers to vectors.
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Initialize the optimizer and model.
print("[INFO] compiling model...")
# Rewriting keras's SGD function for changing its learning rate.
opt = SGD(lr=0.001)
# The categories of Animals dataset is three.
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training
print("[INFO] training network...")
# 32 images will be presented to the network at a time.
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# Evaluate
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), Hypo.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), Hypo.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), Hypo.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), Hypo.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/shallownet_animals_lr0.001.png")
plt.show()
