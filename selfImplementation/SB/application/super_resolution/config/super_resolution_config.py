"""
Created on Sun Jun 17 16:50:00 2018

@author: jercas
"""

import os

# Define the path to the input images we will be using to build the training crops.
# Ukbench100 dataset -- a subset of the larger UKBench dataset.
INPUT_IMAGES_PATH = "ukbench100"

# Define the path to the temporary output dictionaries where will be storing low resolution and high resolution sub-images.
BASE_OUTPUT = "output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# Define the path to the HDF5 file.
INPUTS_DB  = os.path.sep.join([BASE_OUTPUT, "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, "outputs.hdf5"])

# Define the path to the output model file and the plot file.
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srcnn.hdf5"])
PLOT_PATH  = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# Define the batch size and number of epochs for training.
BATCH_SIZE = 128
"""
	  Training for longer can actually hurt performance (where “performance” here is defined as the quality of the output
	super resolution image).
	  Ten epochs should be sufficient for our SRCNN to learn a set of filters to map our low resolution patches to their 
	higher resolution counterparts.
"""
NUM_EPOCHS = 10

# Define the scale (the factor in which we want to learn how to enlarge/upscaling images by) along with the input width
#and height dimensions to SRCNN.
SCALE = 2.0
# The INPUT_DIM is the spatial width and height of our sub-windows (33x33 pixels).
INPUT_DIM = 33

# The label size should be the output spatial dimensions of the SRCNN while our padding ensures we properly crop the label ROI.
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)

# Define the stride which controls the step size of sliding window when creating sub-images.
# Suggests: A stride of 14 pixels for smaller image datasets and a stride of 33 pixels for larger datasets.
STRIDES = 14