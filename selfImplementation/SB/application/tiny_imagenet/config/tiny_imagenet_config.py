"""
Created on Mon Jun 3 10:06:00 2018

@author: jercas
"""

from os import path

# Define the path to the training and validation directories.
TRAIN_PATH = "./dataset/train"
VAL_PATH   = "./dataset/val/images"

# Define the path to the file that map validation filenames to their corresponding class labels(i.e., WordNet-IDs).
VAL_MAPPINGS = "./dataset/val/val_annotations.txt"

# Define the paths to the WordNet hierarchy files which are used to generate class labels from the WordNet-IDs.
WORDNET_IDS = "./dataset/wnids.txt"
WORD_LABELS = "./dataset/words.txt"

# Since there is no access to the testing labels, we need to take a portion of images from the training data and use it instead.
NUM_CLASSES = 200
# 50 images per class(for a total of 10,000 images).
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# Define the path to the output training, validation and testing HDF5 files.
TRAIN_HDF5 = "./dataset/hdf5/train.hdf5"
VAL_HDF5   = "./dataset/hdf5/val.hdf5"
TEST_HDF5  = "./dataset/hdf5/test.hdf5"

# Define the path to the dataset mean which enable to perform the mean normalization.
DATASET_MEAN = "./model/tiny_imagenet_mean.json"

# Define the path to the output directory used for storing plots, classification reports, models, etc.
OUTPUT_DIR = "./output"
MODEL_DIR  = "./model"
MODEL_PATH   = path.sep.join([MODEL_DIR, "checkpoints/deeperGoogLeNet_tinyImageNet.hdf5"])
FIGURE_PATH  = path.sep.join([OUTPUT_DIR, "deeperGoogLeNet_tinyImageNet.png"])
JSON_PATH    = path.sep.join([OUTPUT_DIR, "deeperGoogLeNet_tinyImageNet.json"])

# Training hyper-parameters.
BATCH_SIZE = 64
EPOCHS = 10