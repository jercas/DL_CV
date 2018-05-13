"""
Created on Sat May 12 16:12:00 2018

@author: jercas
"""

from keras.applications import VGG16
import argparse

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", action='store_true', help="whether or not to include top of CNN")
args = vars(ap.parse_args())
print("include_top -- {}".format(args["include_top"]))

# Load the VGG16 network.
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=args["include_top"])

# Loop over the layers in the network and display them to the console for visualization.
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer.__class__.__name__))