"""
Created on Mon Mar 26 16:43:56 2018

@author: jercas
"""

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from preprocessing.SimplePreprocessor import SimplePreprocessor

from imutils import paths

import argparse

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# Grab the list of image paths.
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the image preprocessor, load the dataset from disk and reshape the data matrix(flatten).
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Encode the labels as integers.
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# Loop over dataset of regularizers
for r in (None, "l1", "l2"):
	# Train a SGD classifier using a softmax loss function and the specified regularization function for 10 epochs.
	print("[INFO] training model with {} 'regularization penalty term'".format(r))
	model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42)
	model.fit(trainX, trainY)
	
	# Evaluate the classifier.
	acc = model.score(testX, testY)
	print("[INFO] '{}' regularization penalty accuracy: {:.2f}% \n".format(r, acc*100))