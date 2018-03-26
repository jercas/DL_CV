"""
Created on Wed Mar 22 17:02:45 2018

@author: jercas
"""

import numpy as np
import cv2

CATEGORIES = 3
DIMENSION_SPACE = 3072

# Initialize the class labels and set the seed of the pseudorandom number generator so we can reproduce our results.
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# Randomly initialize our weight matrix and bias vector -- in a 'real' training and classification task, these parameters
#would be 'learned' by our model, but for the sake of this example, instead of using optimization ,we would just use random values.
W = np.random.randn(CATEGORIES, DIMENSION_SPACE)
b = np.random.randn(CATEGORIES)

# Lode example image, resize it and then flatten it into 'feature vector' representation.
orig = cv2.imread("jemma.png")
image = cv2.resize(orig, (32,32)).flatten()

# Compute the hypothesis(prediction output) by taking the dot product between the weight matrix and image pixels,followed
#by adding in the bias.
# F(W,b) = WX + b
hypothesis = W.dot(image) + b

# Loop over the hypothesis + labels and display them.
for (label, hypo) in zip(labels, hypothesis):
	print("[INFO] {}: {:.2f}".format(label, hypo))
# Draw the label with the highest score on the image as prediction.
cv2.putText(orig, "Label: {}".format(labels[np.argmax(hypothesis)]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0.255,0), 2)

# Display input image.
cv2.imshow("Image", orig)
cv2.waitKey(0)