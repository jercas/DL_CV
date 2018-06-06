"""
Created on Wed May 30 16:41:00 2018

@author: jercas
"""

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from SB.nn.conv.minigooglenet import MiniGoogLeNet
from SB.callbacks.trainingMonitor import TrainingMonitor
from PB.polynomial_lr_decay import polynomial_decay
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./model/googlenet_cifar10.hdf5", help="path to save trained model")
ap.add_argument("-o", "--output", default="./output/googlenet_cifar10", help="path to output logs, plots, etc.")
args = vars(ap.parse_args())

# Store the parsed argument.
MODEL_PATH = args["model"]
OUTPUT_PATH = args["output"]

# Initialize the label names of the CIFAR-10 dataset for classification_report() tabulate.
labelNames = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

# Load the training and testing data, then converting the raw images from integers to floats.
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX  = testX.astype("float")

# Apply mean subtraction to the data.
mean = np.mean(trainX, axis=0)
trainX -= mean
testX  -= mean

# Convert the labels from integers to vectors.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.fit_transform(testY)

# Construct the image generator for data augmentation.
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Construct the set of callbacks.
if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)
OUTPUT_PATH = "{}/{}".format(OUTPUT_PATH, "{}".format(os.getpid()))
if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)
figurePath = OUTPUT_PATH
jsonPath = "{}.json".format(OUTPUT_PATH)
callbacks = [TrainingMonitor(figurePath=figurePath, jsonPath=jsonPath), LearningRateScheduler(polynomial_decay)]

# Initialize the optimizer and compile the model.
print("[INFO] compiling model...")
opt = SGD(momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model.
print("[INFO] training model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // 64,
                    epochs=70, callbacks=callbacks, verbose=1)

# Predict the model.
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=64)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)
print(report)

# Serialize report.
path = [OUTPUT_PATH, "model_classification_report.txt"]
f = open(os.path.sep.join(path), "w")
f.write(report)
f.close()

# Compute the raw acc with extra precision.
acc = accuracy_score(testY, predictions)
print("[INFO] score: {}".format(acc))

# Serialize the model.
print("[INFO] serializing model...")
model.save(MODEL_PATH)