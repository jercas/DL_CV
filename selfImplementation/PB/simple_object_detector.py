"""
Created on Mon Jun 11 14:35:00 2018

@author: jercas
"""

from SB.utils.simpy_obj_det import image_pyramid, sliding_window, batch_processing
from imutils.object_detection import non_max_suppression
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import argparse
import time
import cv2

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the being processed input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize variables used for the object detection procedure.
# These are the width and height of input --image. Image is resized, ignoring aspect ratio, to INPUT_SCALE prior to being
#fed through neural network.
INPUT_SIZE = (350, 350)
# The scale of image pyramid.
PYR_SCALE = 1.5
# The step of sliding window.
"""
	The smaller the step, the more windows will be evaluated, and consequently the slower our detector will run. 
	The larger the step, fewer windows will be evaluated and our detector will run faster. 
	There is a trade-off between window size, speed, and accuracy. 
	If your step is too large you may miss detections. 
	If your step is too small, your detector will take a long time to run.
"""
WIN_STEP = 16
# The input ROI size to CNN as if performing classification.
ROI_SIZE = (224, 224)
# The size of the batch to be built and passed through the CNN.
BATCH_SIZE = 64

# Load the network weights from the pre-trained model on ImageNet dataset.
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# Initialize the object detection dictionary which maps class labels to their predicted bounding boxes and associated probability.
labels = {}

# Load the input image from disk and grab its dimensions.
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# Resize the input image to be a square as the shape of INPUT_SIZE.
image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

# Initialize the batch ROIs and (x, y)-coordinates.
batchROIs = None
batchLocs = []

# Start the timer.
print("[INFO] detecting objects...")
start = time.time()

# Loop over every scale of the image generated by the image_pyramid.
for image in image_pyramid(image, scale=PYR_SCALE, minSize=ROI_SIZE):
	# Loop over the sliding window locations to extract each ROI every WIN_STEP.
	for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# Take the ROI and pre-process it then classify the region with the trained classifier/model.
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
		roi = imagenet_utils.preprocess_input(roi)

		# Build our set of batchROIs and batchLocs so we can classify the batch.
		# If the batch is None, then initialize it.
		if batchROIs is None:
			batchROIs = roi
		# Otherwise, add the ROI to the bottom of the batch.
		else:
			batchROIs = np.vstack([batchROIs, roi])
		# Add the (x, y)-coordinates of the sliding window to the batch.
		batchLocs.append((x, y))

		# Check to see if batch is full, if so then apply the batch_preprocessing function to efficiently classify the ROIs.
		if len(batchROIs) == BATCH_SIZE:
			# Classify the batch, then reset the batch ROIs and (x, y)-coordinates.
			labels = batch_processing(model, batchROIs, batchLocs, labels, minProb=args["confidence"])

			# Reset the batch ROIs and (x, y)-coordinates.
			batchROIs = None
			batchLocs = []

# Check to see if there are any remaining ROIs that still need to be classified.
if batchROIs is not None:
	labels = batch_processing(model, batchROIs, batchLocs, labels, minProb=args["confidence"])

# Show how long the detection process took.
end = time.time()
print("[INFO] detections took {:.4f} seconds".format(end - start))

# Loop over the labels dictionary for each of detected objects associated its (x, y)-coordinates in the image.
for key in labels.keys():
	# Clone the input image so we can draw process result information on it.
	clone1 = image.copy()

	# Loop over all bounding boxes and associated probabilities for the label and draw them on the image.
	for (box, prob) in labels[key]:
		(xA, yA, xB, yB) = box
		cv2.rectangle(clone1, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# Show the image *without* apply non-maxima suppression technique.
	cv2.imshow("Without NMS", clone1)

	clone2 = image.copy()
	# Grab the bounding boxes and associated probabilities for each detection, then apply non-maxima suppression to suppress
	#weaker bounding box (i.e., overlapping detections).
	boxes = np.array([result[0] for result in labels[key]])
	proba = np.array([result[1] for result in labels[key]])
	boxes = non_max_suppression(boxes, proba)

	# Loop over the bounding boxes again, this time only drawing the ones that were *not* being suppression.
	for (xA, yA, xB, yB) in boxes:
		cv2.rectangle(clone2, (xA, yA), (xB, yB), (0, 0, 255), 2)

	# Show the image with non-maxima suppression technique.
	print("[INFO] {}: {}".format(key, len(boxes)))
	cv2.imshow("With NMS", clone2)
	cv2.waitKey(0)
