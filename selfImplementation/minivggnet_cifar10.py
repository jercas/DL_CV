"""
Created on Mon Apr 23 11:33:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import matplotlib
# Set matplotlib backend to Agg to indicate to create a non-interactive figure that will simply be saved to disk.
# Why? ^^^
# Depending on what your default matplotlib backend is and whether you are accessing your deep learning machine remotely
#(via SSH, for instance), X11 session may timeout. If that happens, matplotlib will error out when it tries to display figure.
matplotlib.use("Agg")

from callbacks.trainingMonitor import TrainingMonitor
from nn.conv.minivggnet import MiniVGGNet
from stepBased_lr_decay import stepBased_decay
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
ap.add_argument("-m", "--monitor", required=False, action="store_true",
                help="decide whether to use training monitor which can plot loss curve at the end of every epoch.")
ap.add_argument("-c", "--checkpoint", required=False,
                help="decide whether to store checkpoint which can serialized models during the training process on each improvement epoch.")
args = vars(ap.parse_args())

# Show information on the process ID.
print("[INFO] process ID: {}".format(os.getpid()))

# Grab the MNIST dataset, pre-segmented into training and testing split(50000:10000 images, per 5000:1000 a class).
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# Convert the data type from unsigned 8-bit integers to floating point, followed by scale the RGB pixels to the range [0, 1.0].
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

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

# Initialize the optimizer and model.
print("[INFO] compiling model...")
"""
# Time-based decay: slowly reduce the learning rate over time, common setting for decay is to divide the initial lr
#by total number of epochs.(here 0.01/40)
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
"""

# Step-based decay: alpha = initialAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
# lr parameter can be leave out entirely since it is using the LearningRateScheduler callback.
opt = SGD(momentum=0.9, nesterov=True)

# Define the set of callbacks function to be passed to the model during training.
# Keras will call callbacks at the start or end of every epoch, mini-batch update, etc.
# Then 'LearningRateSchedular' will call 'stepBased_decay' at the end of every epoch, decide whether to update learning 
#rate prior to the next epoch starting. 
callbacks = [LearningRateScheduler(stepBased_decay)]

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

if args["monitor"]:
	if not os.path.exists("{}".format(args["output"])):
		os.mkdir("{}".format(args["output"]))
	if not os.path.exists("{}/{}".format(args["output"], os.getpid())):
		os.mkdir("{}/{}".format(args["output"], os.getpid()))

	print("\n[INFO] monitor module establish!")
	figurePath = "{}/{}".format(args["output"], os.getpid())
	jsonPath   = "{}/{}/{}.json".format(args["output"], os.getpid(), os.getpid())
	# Construct the set of callbacks.
	callbacks.append(TrainingMonitor(figurePath=figurePath, jsonPath=jsonPath))

if args["checkpoint"]:
	if not os.path.exists("{}".format(args["checkpoint"])):
		os.mkdir("{}".format(args["checkpoint"]))
	if not os.path.exists("{}/{}".format(args["checkpoint"], os.getpid())):
		os.mkdir("{}/{}".format(args["checkpoint"], os.getpid()))

	print("\n[INFO] checkpoint module establish!\n")
	# A template string value that keras uses when writing checkpoing-models to disk based on its epoch and the validation
	#value on the current epoch.
	fname = os.path.join("{}/{}/".format(args["checkpoint"], os.getpid()), "checkpoint-{epoch:03d}-{val_loss:.4f}.hd5f")
	# monitor -- what metric would like to monitor;
	# mode -- controls whether the ModelCheckpoint be looking for values that minimize metric or maximize it in the contrary.
	#         such as, if you monitor val_loss, you would like to minimize it and if monitor equals to val_acc then you should maximize it.
	# save_best_only -- ensures the latest best model (according to the metric monitored) will not be overwritten.
	# verbose=1 -- simply logs a notification to terminal when a model is being serialized to disk during training.
	# period -- the interval epochs between two saved checkpoints.
	checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
	# Construct the set of callbacks.
	callbacks.append(checkpoint)

# Train.
print("[INFO] training network...")
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1, callbacks=callbacks)

# Evaluate.
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), Hypo.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), Hypo.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), Hypo.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), Hypo.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("{}.png".format(args["output"]))
plt.show()
