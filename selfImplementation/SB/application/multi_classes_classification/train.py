"""
Created on Tue May 22 12:31:00 2018

@author: jercas
"""

# Set the matplotlib bakcend so figures can be saved in the background without showing.
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from SB.nn.conv.smallvggnet import SmallVGGNet

import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="./dataset", help="path to input data (i.e., directory of images)")
ap.add_argument("-l", "--labelbin", type=str, default="./model/lb.pickle", help="path to output label binarizer")
ap.add_argument("-m", "--model", type=str, default="./model/pokedex.hdf5", help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, default="./output/plot.png", help="path to output acc/loss curve.")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for, initialize learning rate, batch size, and image dimensions.
EPOCHS = 100
INIT_LR = 1e-3          # The default value for the 'Adam' optimizer.
BATCH_SIZE = 32
IMAGE_DIMS = (96, 96, 3)

# Initialize the data and labels.
data = []
labels = []

# Grob the image paths and randomly shuffle them.
print("[INFO] loading image...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# Loop over the input image.
for imagePath in imagePaths:
	# Load image, preprocess it, and store it in the data list.
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# Extract set of class labels (each element is a 2-element list, e.g., ['red', 'dress']) from the image path and update the label list.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Scaling the raw pixel intensities to the range [0, 1].
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix:{}, image ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

print("[INFO] class labels:")
# Because out dataset contains multiple(two) categories label, so we cannot use the standard LabelBinarizer class for multi-class classification.
lb = LabelBinarizer()
# Convert single integer to a two-hot encode vector.
labels = lb.fit_transform(labels)
# Loop over each of the possible class labels and show them.
for (i, label) in enumerate(lb.classes_):
	print("{}. {}.".format(i + 1, label))

# Partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Construct the image generator for data augmentation.
aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Initialize the model.
print("[INFO] compiling model...")
model = SmallVGGNet.build(width=IMAGE_DIMS[1],
                          height=IMAGE_DIMS[0],
                          depth=IMAGE_DIMS[2],
                          classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Compile and train.
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
History = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                              validation_data=(testX, testY),
                              steps_per_epoch=len(trainX) // BATCH_SIZE,
                              epochs=EPOCHS, shuffle=True, verbose=1)

# Serialize the model to disk.
print("[INFO] Serialize trained model...")
model.save(args["model"])
# Save the multi-label binarizer to disk.
print("[INFO] Save label binarizer...")
f = open(args["labelbin"], 'wb')
f.write(pickle.dumps(lb))
f.close()

# Evaluate model.
predictions = model.predict(testX, batch_size=64, verbose=1)
report = classification_report(testY.argmax(axis=1),
                               predictions.argmax(axis=1),
                               target_names=lb.classes_)
print(report)
# Serialize report.
f = open("./output/report.txt", "w")
f.write(report)
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), History.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])