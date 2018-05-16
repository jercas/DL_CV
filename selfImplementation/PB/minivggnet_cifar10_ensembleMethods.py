"""
Created on Tue May 15 11:51:00 2018

@author: jercas
"""

# Set the matplotlib backend so that figures can be saved in the background without visualizing/showing.
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from SB.nn.conv.minivggnet import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Constrcut the argument parse and parse arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output figures directory")
ap.add_argument("-m", "--model", required=True, help="path to output models directory")
ap.add_argument("-n", "--num_models", type=int, default=5, help="# of models to train")
args = vars(ap.parse_args())

# Load the training and testing data, then scale it into the range [0, 1].
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# Convert the labels from integers to vectors.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
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

# Construct the image generator for data augmentation.
aug = ImageDataGenerator(rotation_range=10,             # Randomly rotated +/- 30 degrees; "typical[10, 30]"
                         width_shift_range=0.1,             # Horizontally shifted by a factor of 0.1; "typical[0.1, 0.2]"
                         height_shift_range=0.1,            # Vertically shifted by a factor of 0.1; "typical[0.1, 0.2]"
                         horizontal_flip=True,              # Randomly horizontally flipped, Unless flipping your image changes the class label,
                         fill_mode="nearest")               #You should always include horizontal flipping as well.

# Loop over the number of models to train.
for i in np.arange(0, args["num_models"]):
	# Initialize the optimizer and model.
	print("[INFO] training model {}/{}".format(i+1, args["num_models"]))
	opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)

	# Initialize model architecture.
	model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(labelNames))
	# Build model.
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	# Train model.
	History = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
	                              validation_data=(testX, testY), epochs=40,
	                              steps_per_epoch=len(trainX)//64, verbose=1)

	# Serialize model.
	path = [args["model"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(path))

	# Evaluate model.
	predictions = model.predict(testX, batch_size=64, verbose=1)
	report = classification_report(testY.argmax(axis=1),
	                               predictions.argmax(axis=1),
	                               target_names=labelNames)

	# Serialize report.
	path = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(path), "w")
	f.write(report)
	f.close()

	# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, 40), History.history["loss"], label="train_loss")
	plt.plot(np.arange(0, 40), History.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, 40), History.history["acc"], label="train_acc")
	plt.plot(np.arange(0, 40), History.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy for model {}".format(i))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()

	# Serialize plot.
	path = [args["output"], "model_{}.png".format(i)]
	plt.savefig(os.path.sep.join(path))
	plt.close()