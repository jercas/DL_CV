"""
Created on Tue May 16 13:17:00 2018

@author: jercas
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# Construct the argument parse and parse arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to models directory")
args = vars(ap.parse_args())

# Load the testing data, then scale it into the range [0, 1].
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# Convert the labels from integers to vectors.
lb = LabelBinarizer()
testY  = lb.fit_transform(testY)

# Initialize the label names for the CIFAR-10 dataset
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

# Construct the path used to collect the models then initialize the models list.
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []

# Loop over the model paths, loading the model, and adding it to the list of models.
for (i, modelPath) in enumerate(modelPaths):
	print("[INFO] loading model {}/{}".format(i+1, len(modelPaths)))
	models.append(load_model(modelPath))

# Initialize the list of predictions.
print("[INFO] evaluating ensemble...")
predictions = []

# Loop over the models.
for model in models:
	# Use the current model to make predictions on the testing data then store these predictions in the aggregate predictions list.
	predictions.append(model.predict(testX, batch_size=64))

# Average the probabilities across all model predictions, then show a classification report.
# (5, 10000, 10) --avg--> (1000, 10).
ensembledPredictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
                            ensembledPredictions.argmax(axis=1),
                            target_names=labelNames))
