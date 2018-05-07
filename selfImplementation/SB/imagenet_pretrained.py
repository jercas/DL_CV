"""
Created on Thu Apr 26 10:46:00 2018

@author: jercas
"""

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
# imagenet_utils module give ability to access to the imagenet sub-module, a handy set of convenience function that
#will make pre-processing input images and decoding output classification eaiser.
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
args = vars(ap.parse_args())

# Define a dictionary that maps model names to its classes inside keras individual.
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception,
	"resnet": ResNet50
}

# Ensure a valid model name was supplied via command line argument.
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should be a key in the 'MODELS' dictionary which include VGG16, VGG19, 1ncepthonV3, Xception, Resnet50")

# Initialize the input image shape (224*224 pixels） along with the pre-processing function.
#  VGG16, VGG19 and ResNet all accept 224*224 input images while InceptionV3 and Xception require 229*229 pixel inputs.
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	# Updating preprocess to use a separate pre-processing function that performs a different type of scaling since the
	#Inception and its extension Xception is act as "multi-level feature extractor" by computing different size convolution
	#filters within the same module of the network.
	preprocess = preprocess_input

# Load network weights from disk.
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
# Adaptive function call.
model = Network(weights="imagenet")

# Load the input image using the keras helper utility while ensuring the image is resized to 'inoutShape', the required
#input dimensions for ImageNet pre-trained network.
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# Input image is now represented as a NumPy array of shape(inputShape[0]-width, inputShape[1]-height, 3-depth).
# Then expand the shape's dimension to (1-batch_size(when predict phase, it equals to 1),inputShape[0]-width, inputShape[1]-height, 3-depth).
image = np.expand_dims(image, axis=0)

# Pre-process the image using the appropriate function based on the model that choose to been loaded.(i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# Classify the image.
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
# imagenet_utils.decode_predictions(preds): Returns a list of ImageNet class label IDs, labels and the probability associated with each class label.
P = imagenet_utils.decode_predictions(preds)

# Loop over the predictions and display the rank-5 predictions + probabilities to terminal.
for (i, (imagenet_ID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i+1, label, prob*100))

# Load the image via openCV, draw the top prediction on the image and display the image to screen.
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)