"""
Created on Wed Jun 6 15:44:00 2018

@author: jercas
"""

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from nn.conv.resnet import ResNet
from callbacks.epochCheckPoint import EpochCheckpoint
from callbacks.trainingMonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", type=str, default="./model/resnet_cifar10/checkpoints", help="path to output checkpoint directory")
ap.add_argument("-o", "--output", type=str, default="./output/resnet_cifar10/")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# Store the parsed argument.
CHECKPOINTS_PATH = args["checkpoint"]
OUTPUT_PATH = args["output"]
MODEL_PATH = args["model"]
START_EPOCH =  args["start_epoch"]

# Define the hyper-parameters.
BATCH_SIZE = 128
EPOCHS = 100

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

# If there is no specific model checkpoint supplied, then initialize the network (ResNet-56) and compile the model.
if MODEL_PATH is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=1e-1)
	# stages: This tuple indicates that we will be learning three stages with each stage containing nine residual modules stacked on top of each other.
	#       In between each stage, we will apply an additional residual module to decrease the volume size.
	# filters: The number of filters that the CONV layers will learn in each stage.
	#       The first CONV layer (before any residual model is applied) will learn K = 64 filters.
	#       The remaining entries, 64, 128, and 256 correspond to the number of filters each of the residual module stages will learn.
	model = ResNet.build(width=32, height=32, depth=3, classes=len(labelNames),
	                     stages=(9, 9, 9), filters=(64, 64, 128, 256), reg=5e-4)

# Otherwise, load the checkpoint from disk.
else:
	print("[INFO] loading:{}...".format(MODEL_PATH))
	model = load_model(MODEL_PATH)

	# Update the learning rate.
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# Construct the set of callbacks.
if not os.path.exists(CHECKPOINTS_PATH):
	os.mkdir(CHECKPOINTS_PATH)
CHECKPOINTS_PATH = "{}/{}".format(CHECKPOINTS_PATH, "{}".format(os.getpid()))

if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)
figurePath = "{}/{}.json".format(OUTPUT_PATH, os.getpid())
jsonPath = "{}/{}.json".format(OUTPUT_PATH, os.getpid())
callbacks = [TrainingMonitor(figurePath=figurePath, jsonPath=jsonPath, startAt=START_EPOCH),
             EpochCheckpoint(CHECKPOINTS_PATH, every=5, startAt=START_EPOCH)]

# Train the network.
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // BATCH_SIZE,
                    epochs=EPOCHS, callbacks=callbacks, verbose=1)

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