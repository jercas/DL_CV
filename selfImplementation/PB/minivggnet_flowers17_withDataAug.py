"""
Created on Mon Apr 7 15:56:00 2018

@author: jercas
"""

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.utils import to_categorical, np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, sparse_categorical_accuracy

from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from AspectAwarePreprocessor import AspectAwarePreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
from conv.minivggnet import MiniVGGNet

import matplotlib.pyplot as plt
from imutils import paths
import glob
import numpy as np
import argparse
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../../sourceCode/practitionerBundle/PBCode/datasets/flowers17/images",
                help="path to input dataset")
ap.add_argument("-a", "--augment", action='store_true', help="whether to use data augmentaion")
args = vars(ap.parse_args())

CLASSES = len(glob.glob("{}/*".format(args["dataset"])))
# Grab the list of images that we'll be describing, then extract the class label names from the image paths.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Initialize the image preprocessors.
# Resize image to 64 * 64 for input.
aap = AspectAwarePreprocessor(64, 64)
# Convert image to Keras-compatible arrays.
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel intensities to the range [0, 1] by dividing the raw pixel intensities by 255.
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Convert the labels from integers to vectors.
# Keras.utils.to_categorical() can just transform integer label to one-hot code array.
# Keras.utils.to_categorical() has the same effect and return as keras.utils.np_utils.to_categorical().
# labels = to_categorical(labels, CLASSES)

# sklearn.preprocessing.LabelEncoder() can transform string label to a single number, so we next call keras.utils.np_utils.to_categorical() to transform each
#single number to a one-hot code array.
# le = LabelEncoder()
# strToNum = le.fit_trainsform(labels)
# labels = np_utils.to_categorical(strToNum, CLASSES)

# sklearn.preprocessing.LabelBinary() can directly transform string label to one-hot code array.
lb =LabelBinarizer()
labels = lb.fit_transform(labels)

# Partition the data into training and testing splits as 3:1.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels ,random_state=42)

# Initialize the optimizer and model.
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy", "categorical_accuracy", "sparse_categorical_accuracy"])

if args["augment"]:
	print("[INFO] data augmentation mode...")
	aug = ImageDataGenerator(rotation_range=30,             # Randomly rotated +/- 30 degrees; "typical[10, 30]"
                         width_shift_range=0.1,             # Horizontally shifted by a factor of 0.1; "typical[0.1, 0.2]"
                         height_shift_range=0.1,            # Vertically shifted by a factor of 0.1; "typical[0.1, 0.2]"
                         shear_range=0.2,                   # Sheared by 0.2;
                         zoom_range=0.2,                    # Zoomed by uniformly sampling in the range [0.8, 1.2];
                         horizontal_flip=True,              # Randomly horizontally flipped, Unless flipping your image changes the class label,
                         fill_mode="nearest")               #You should always include horizontal flipping as well.
	# Training model.
	print("[INFO] training model...")
	History = model.fit_generator(  # Only apply data augmentayion to the training set.
									aug.flow(trainX, trainY, batch_size=32),    # Used to generate new training samples from training data.
									validation_data=(testX, testY),
									steps_per_epoch=len(trainX) // 32,          # Controls the number of batches per epoch.
																				# Typical determine this parameter's value by dividing the total number of
																				#training samples by batch size and then casting it to an integer(//).
									epochs=100, verbose=1)
	dataAug = "with_DataAug"

else:
	# Training model.
	print("[INFO] training model...")
	History = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)
	dataAug = "withOut_DataAug"

# Evaluating Phase:
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
# argmax() -- determine the class with the largest probability, which return the index of the class label with largest probability.
# sklearn.classification_report() -- tabulating the final output/predict classification by the trained network
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

model.save("./model/minivggnet_flowers17_{}.hdf5".format(dataAug))
# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), History.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/minivggnet_flower17_{}".format(dataAug))
plt.show()