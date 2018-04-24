"""
Created on Mon Apr 23 11:33:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler

import matplotlib
# Set matplotlib backend to Agg to indicate to create a non-interactive figure that will simply be saved to disk.
# Why? ^^^
# Depending on what your default matplotlib backend is and whether you are accessing your deep learning machine remotely
#(via SSH, for instance), X11 session may timeout. If that happens, matplotlib will error out when it tries to display figure.
matplotlib.use("Agg")

from nn.conv.minivggnet import MiniVGGNet
from stepBased_lr_decay import stepBased_decay
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

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
callbacks = [LearningRateSchedular(stepBased_decay)]
"""
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# Train.
print("[INFO] training network...")
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)
"""
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1, callbacks=callbacks)
"""

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
plt.savefig(args["output"])
plt.show()
