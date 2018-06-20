"""
Created on Tue Jun 19 14:31:00 2018

@author: jercas
"""

from super_resolution.config import super_resolution_config as config
from keras.models import load_model
from scipy import misc
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image/low resolution image")
ap.add_argument("-b", "--baseline", required=True, help="path to baseline image")
ap.add_argument("-o", "--output", required=True, help="path to output image/after apply super resolution image")
args = vars(ap.parse_args())

# Load the pre-trained model.
model = load_model(config.MODEL_PATH)
# Load the input image, then grab the dimensions of the input image and crop the image such that it tiles nicely when
#applying sliding window and passing the sub-images through the SRCNN.
print("[INFO] loading image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

"""
	Upscaling our image by a factor of SCALE serves two purposes:
	1. It gives us a baseline of what standard upsizing will look like using traditional image processing.
	2. Our SRCNN requires a high resolution input of the original low resolution image — this scaled image serves that purpose.
"""
# Resize the input image using bicubic interpolationhen write the baseline image to disk.
scaled = misc.imresize(image, config.SCALE / 1.0, interp="bicubic")
cv2.imwrite(args["baseline"], scaled)

# Allocate memory for the output image.
output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]

# Slide a window from left-to-right and top-to-bottom.
"""
	For each stop along the way, in LABEL_SIZE steps, we crop out the sub-image from scaled.
	The spatial dimensions of crop match the input dimensions required by our SRCNN.
	Then take the crop sub-image and pass it through our SRCNN for inference.
	The output of the SRCNN, P has spatial dimensions LABEL_SIZE x LABEL_SIZE x CHANNELS.
	Finally, store the high resolution prediction from the network in the output image.
"""
for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
	for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
		# Crop the ROI from scaled image.
		crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]
		crop = np.expand_dims(crop, axis=0)
		# Make a prediction on the crop and store it in output image.
		P = model.predict(crop)
		P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
		output[y + config.PAD:y + config.PAD + config.LABEL_SIZE, x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P

# Remove any of the black borders in the output image caused by the padding, then clip any values that fall outside the range [0, 255].
output = output[config.PAD:h - ((h % config.INPUT_DIM) + config.PAD), config.PAD:w - ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

# Write the output image to disk.
cv2.imwrite(args["output"], output)