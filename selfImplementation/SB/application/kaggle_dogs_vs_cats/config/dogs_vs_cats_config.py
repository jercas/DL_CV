"""
Created on Thu May 24 16:05:00 2018

@author: jercas
"""

# Define the paths to the image directory.
IMAGES_PATH = "./dataset/train"

# Since we don't have validation data or access to the testing labels, we need to take a number of images from training set
#as validation set.
NUM_CLASSES = 2
NUM_VAL_CLASSES = 1250 * NUM_CLASSES
NUM_TEST_CLASSES = 1250 * NUM_CLASSES

# Define the path to the output training, validation, and testing HDF5 files/output serialized weights.
TRAIN_HDF5 = "./dataset/hdf5/train.hdf5"
VAL_HDF5 = "./dataset/hdf5/val.hdf5"
TEST_HDF5 = "./dataset/hdf5/test.hdf5"

# Define the path to the output model file.
MODEL_PATH = "./model/alexnet_dogs_vs_cats.model"
# Define the path to the dataset mean.
# Used to store the average red, green, and blue pixel intensity values across the entire (training) dataset.
DATASET_MEAN = "./model/dogs_vs_cats_mean.json"
# Define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "./output"
CHECKPOINT_PATH  = "./model"

# Training Hyper-parameters.
BATCH_SIZE = 64
EPOCHS = 75