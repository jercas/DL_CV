"""
Created on Fri Apr 20 17:10:00 2018

@author: jercas
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from nn.conv.lenet import LeNet
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Grab the MNIST dataset.
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data

# If using "channels first" ordering, then reshape the design matrix such that the matrix is: num_samples * depth * rows * columns.
if K.image_data_format() == "channels_first":
	data = data.reshape(data.shape[0], 1, 28, 28)

# Otherwise, using "channels last" ordering, then reshape the design matrix shape should be: num_samples * rows * columns * depth.
else:
	data = data.reshape(data.shape[0], 28, 28, 1)

# Scale the input data to the range [0, 1] and perform a train/test split.
(trainX, testX, trainY, testY) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)
# One-hot coding: Convert the labels from integers to vectors.
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

# Initialize the model.
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train.
print("[INFO] training network...")
Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=50, verbose=1)

# Evaluate.
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=['零','壹','贰','叁','肆','伍','陆','柒','捌','玖']))

# Plot the training loss, training accuracy, validation loss, validation accuracy over time.
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), Hypo.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), Hypo.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), Hypo.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), Hypo.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/lenet_mnist_lr0.005.png")
plt.show()