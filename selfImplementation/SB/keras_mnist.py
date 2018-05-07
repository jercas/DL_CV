"""
Created on Sun Apr 9 11:39:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Sequential class indicates that network model will be feed-forward and layers will be added to the class sequentially, one on top of the other.
from keras.models import  Sequential
# Dense class indicates full-connected layers.
from keras.layers.core import Dense
# Stochastic Gradient Descend.
from keras.optimizers import SGD
from keras.activations import sigmoid, softmax
from keras.losses import categorical_crossentropy

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import argparse

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
ic(args)

# Grab the MNIST dataset.
dataset = datasets.fetch_mldata("MNIST Original")
# Scale the raw pixel intensities to the range [0, 1.0].
data = dataset.data.astype("float") / 255.0
# Construct the training and testing splits.
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)
# Convert the labels from integers to vectors.
lb = LabelBinarizer()
ic(trainY)
trainY = lb.fit_transform(trainY)
ic(trainY)
testY = lb.fit_transform(testY)

# Define the 784-dim(28*28 pixels)-256-128-10 architecture using keras.
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation=sigmoid))
model.add(Dense(128, activation=sigmoid))
model.add(Dense(10, activation=softmax))

# Training Phase:
print("[INFO] training network...")
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01), metrics=["accuracy"])
# Being a bit lenient that just use test set as validation set at here, since simply focus on how to train a nn from scratch using Keras.
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# Evaluating Phase:
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
# argmax() -- determine the class with the largest probability, which return the index of the class label with largest probability.
# sklearn.classification_report() -- tabulating the final output/predict classification by the trained network
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

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
plt.savefig(args["output"])
plt.show()
