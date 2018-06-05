"""
Created on Tue Jun 5 10:24:00 2018

@author: jercas
"""

import matplotlib
matplotlib.use("Agg")
import tiny_imagenet_config as config
from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from preprocessing.SimplePreprocessor import SimplePreprocessor
from preprocessing.MeanPreprocessor import MeanPreprocessor
from preprocessing.PatchPreprocessor import PatchPreprocessor
from callbacks.trainingMonitor import TrainingMonitor
from callbacks.epochCheckPoint import EpochCheckpoint
from io.hdf5DatasetReader import HDF5DatasetReader
from nn.conv.deepergoolenet import DeeperGoogLeNet

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import json

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", type=str, default="./model/checkpoints", help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training set")
args = vars(ap.parse_args())

# Construct the training image generator for data augmentation.
aug = ImageDataGenerator(rotation_range=18,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
						 zoom_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Load the RGB means for the training set preprocessing -- mean subtraction and normalization.
means = json.loads(open(config.DATASET_MEAN).read())
# Initialize the image preprocessors.
pp  = PatchPreprocessor(width=64, height=64)
sp  = SimplePreprocessor(width=64, height=64)
mp  = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators.
trainGen = HDF5DatasetReader(dbPath=config.TRAIN_HDF5,
                             batchSize=config.BATCH_SIZE,
                             preprocessors=[pp, mp, iap],
                             aug=aug,
                             classes=config.NUM_CLASSES)
valGen   = HDF5DatasetReader(dbPath=config.VAL_HDF5,
                             batchSize=config.BATCH_SIZE,
                             preprocessors=[sp, mp, iap],
                             classes=config.NUM_CLASSES)

# If there is no specific model checkpoint supplied, then initialize the network and compile the model.
if args["model"] is None:
	print("[INFO] compiling model...")
	model = DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=2e-4)
	opt = Adam(lr=1e-3)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Otherwise, load the *specific* checkpoint for disk.
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# Update the learning rate.
	print("[INFO] old learning rate α: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate α: {}".format(K.get_value(model.optimizer.lr)))

# Construct the set of callbacks.
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
	TrainingMonitor(figurePath= config.FIGURE_PATH, jsonPath=config.JSON_PATH, startAt=args["start_epoch"])]

# Train the network architecture.
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=config.BATCH_SIZE * 2,
	callbacks=callbacks,
	verbose=1)

# Close the database.
trainGen.close()
valGen.close()