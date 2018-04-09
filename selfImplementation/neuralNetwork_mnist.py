"""
Created on Sun Apr 8 17:43:00 2018

@author: jercas
"""

from nn.neuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from icecream import ic


# Load the MNIST dataset and apply min/max scaling to scale the pixel intensity value to the range [0,1](each image is
#represented by an 8*8 = 64-dim feature vector).
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")

# feature scaling: min/max normalizing.
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# Construct the training and testing splits.
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)
ic(trainY)
# Convert the labels from integers to vectors, one-hot encoding.
trainY = LabelBinarizer().fit_transform(trainY)
ic(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Train the network.
print("[INFO] training network...")
# 64(8*8 pixels)-32-16-10 architecture.
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# Evaluate the trained network.
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
# argmax() -- return the index of the label with the highest predicted probability.
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=['零','壹','贰','叁','肆','伍','陆','柒','捌','玖']))

