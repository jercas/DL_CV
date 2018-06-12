"""
Created on Tue Jun 12 13:45:00 2018

@author: jercas
"""

from SB.nn.conv.neuralStyle import  NeuralStyle
from keras.applications import VGG19

# Initialize the settings dictionary which contains some parameters.
SETTINGS = {
	# Initialize the path to the input (i.e., content) image, style image, and the path to output dictionary.
	"input_path": "input/jp.jpg",
	"style_path": "input/starry_night.jpg",
	"output_path": "output/style_transfer",

	# Define the CNN to be used for style transfer, along with the set of content layer and style layers, respectively.
	"net": VGG19,
	"content_layer": "block4_conv2",
	"style_layers": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],

	# Store the content, style, and total-variation weights, α, β and γ, respectively.
	"content_weight": 1.0,
	"style_weight": 100.0,
	"tv_weight":  10.0,

	# Number of iterations.
	"iterations": 50,
}

# Perform neural style transfer.
print("[INFO] style transfer start...")
ns = NeuralStyle(SETTINGS)
ns.transfer()
