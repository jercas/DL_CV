"""
Created on Fri Apr 20 15:15:00 2018

@author: jercas
"""
from preprocessing.SimplePreprocessor import SimplePreprocessor
from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from datasets.SimpleDatasetLoader import SimpleDatasetLoader

from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input datasets")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

# Initialize the class labels.
classLabels = ["cat", "dog", "panda"]

# Grab the list of images in the dataset then randomly sample indexes into the image path list.
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# Initialize the image preprocessors.(Preprocessing the test set as training procedure in the previous)
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel grey intensities to the range [0, 1].
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# Load the pre-trained model.
print("[INFO] loading pre-trained model")
model = load_model(args["model"])

# Predict
print("[INFO] predicting")
# .predict() return a list of probabilities for every images in data - one probability for each class label, respectively.
# .argmax() on axis=1 finds the index of the class label with the largest probability for each image.
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Loop over the sample images.
for (i, imagePath) in enumerate(imagePaths):
	# Load the test example image, draw the prediction, and display it to screen.
	image = cv2.imread(imagePath)
	cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)