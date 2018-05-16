"""
Created on Sun May 13 11:35:00 2018

@author: jercas
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from SB.preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SB.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor
from SB.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from SB.nn.conv.fcHeadNet import FCHeadNet

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# Construct the image generator for data augmentation.
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Grab the list of images that we'll be describing, then extract the class label names from the image paths.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = [str(className) for className in np.unique(classNames)]

# Initialize the image preprocessors.
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel intensities to the range [0, 1].
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Dataset split and one-hot encoding.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.fit_transform(testY)

# Load the VGG16 network, ensuring the head FC layer sets are left off(no-top model).
baseModel = VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))
# Initialize the new head of the network, a set of FC layers followed by a softmax classifier.
headModel = FCHeadNet.build(baseModel=baseModel, classes=len(classNames), D=256)
# Place the new head FC model **on top of** the base model -- combine these two architecture as our actual train model.
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze stage: Loop over all original layers(i.e., layers in the base model) and **freeze** them so they will **not be**
#updated during the training process which indicates the already learned feature won't be destroy by BP through the whole architecture.
for layer in baseModel.layers:
	layer.trainable = False

# Compile the model (this needs to be done after setting original layers to being non-trainable).
print("[INFO] compiling model...")
# Using a small learning rate of 1e-3 to warm ip the new FC head.
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network for a few epochs (all other original layers are frozen) -- this will allow the new FC
#layers to start to become initialized with actual **learned values** versus pure random.
print("[INFO] training head...")
# Each image is being fully forward propagated and the gradients are only being partially backpropagated(end after the FC layers).
# Only "warm up"(about 10-30 epochs) the head and not change the weights in the body of network.
History_warmUp = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                           validation_data=(testX, testY), epochs=25, steps_per_epoch=len(trainX) // 32, verbose=1)

# Evaluating the network after warm up for comparing the effects of fine-tuning before and after allowing the head to warm up.
print("[INFO] evaluating after warm up the FC head...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
"""
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 25), History_warmUp.history["loss"], label="train_loss")
plt.plot(np.arange(0, 25), History_warmUp.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 25), History_warmUp.history["acc"], label="train_acc")
plt.plot(np.arange(0, 25), History_warmUp.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/finetune_flower17_warm_up.jpg")
plt.show()
"""
# Now that the FC layers have been warm up/initialized, lets unfreeze the final set of CONV layers and make them trainable.
# First, only unfreezing the top CONV layers and then continuing training, if ACC continues to improve (without overfitting),
#you may want to consider unfreezing more layers in the original body.
for layer in baseModel.layers[15:]:
	layer.trainable = True

# For the changes to the model to take affect we need to re-compile the model and switch to using SGD with a **very small* lr.
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training the model again, this time fine-tuning **both** the final set of CONV layers along with set of new FC layers.
History_fineTuned = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=100, steps_per_epoch=len(trainX) // 32, verbose=1)

# Evaluate the network on the fine-tuned model.
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# Save the model to disk.
print("[INFO] serializing model...")
model.save(args["model"])

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), History_fineTuned.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), History_fineTuned.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), History_fineTuned.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), History_fineTuned.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/finetune_flowers17_fine_tuned.jpg")
plt.show()