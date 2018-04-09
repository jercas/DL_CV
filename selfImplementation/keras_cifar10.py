"""
Created on Sun Apr 9 17:15:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Sequential class indicates that network model will be feed-forward and layers will be added to the class sequentially, one on top of the other.
from keras.models import  Sequential
# Dense class indicates full-connected layers.
from keras.layers.core import Dense
# Stochastic Gradient Descend.
from keras.optimizers import SGD
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import argparse

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
ic(args)

# Grab the MNIST dataset, pre-segmented into training and testing split(50000:10000 images, per 5000:1000 a class).
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# Convert the data type from unsigned 8-bit integers to floating point, followed by scale the RGB pixels to the range [0, 1.0].
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# Reshape design matrix(flatten to 1-dim), 3072-dim = (32*32)pixels*3channels.
# trainX.shape() - from (50000, 32, 32, 3) to (50000, 3072)
trainX = trainX.reshape((trainX.shape[0], 3072))
# testX.shape() - from (10000, 32, 32, 3) to (10000, 3072)
testX = testX.reshape((testX.shape[0], 3072))
# One-hot encoding: convert the labels from integers to vectors.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
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

# Define the 3072-dim((32*32)pixels * 3channels)-1024-512-10 architecture using keras.
model = Sequential()
# Swap out the antiquated sigmoid activation for a ReLU activation in hopes of improve network performance as other researchers always do in state-of-art.
model.add(Dense(1024, input_shape=(3072,), activation=relu))
model.add(Dense(512, activation=relu))
model.add(Dense(10, activation=softmax))

# Training Phase:
print("[INFO] training network...")
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01), metrics=["accuracy"])
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# Evaluating Phase:
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
# argmax() -- determine the class with the largest probability, which return the index of the class label with largest probability.
# sklearn.classification_report() -- tabulating the final output/predict classification by the trained network
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

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
