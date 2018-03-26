# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:02:45 2018

@author: jercas
"""

from sklearn.neighbors          import KNeighborsClassifier
from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import classification_report

from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from preprocessing.SimplePreprocessor import SimplePreprocessor

from imutils import paths

import argparse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# Grab the list of images that we'll be describing
print("[INFO] loding images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the image preprocessor, load the dataset from disk and reshape the data matrix.
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
# Data:3000 images of 3-dimension 32*32*3 pixels, labels:vector
(data, labels) = sdl.load(imagePaths, verbose=500)
# Flatten from 3D representation to a single list of pixel intensities. From 3000*3-dim(32*32*3) to 3000*3072(32*32*3)-dim.
data = data.reshape((data.shape[0], 3072))

# Show some information on memory consumption of the images.
print("[INFO] features matrix {:.1f}MB".format(data.nbytes / (1024*1000.0)))

# Encode the labels as integers(convert labels from representation strings to integers), one unique integer per class.
# cat - 0, dog - 1,panda - 2.
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing splits using 75% of data for training and the remaining 25% as test set.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate a k-NN classifier on the raw pixel intensities.
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))