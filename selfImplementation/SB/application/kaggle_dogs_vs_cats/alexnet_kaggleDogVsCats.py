"""
Created on Mon May 28 13:45:00 2018

@author: jercas
"""

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from SB.application.kaggle_dogs_vs_cats.config import dogs_vs_cats_config as config
from SB.preprocessing.MeanPreprocessor import MeanPreprocessor
from SB.preprocessing.PatchPreprocessor import PatchPreprocessor
from SB.preprocessing.SimplePreprocessor import SimplePreprocessor
from SB.preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SB.callbacks.trainingMonitor import TrainingMonitor
from SB.io.hdf5DatasetReader import HDF5DatasetReader
from SB.nn.conv.alexnet import AlexNet

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import json
import os

# Construct the training image generator for data augmentation.
aug = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.15,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Load the RGB means for the training set.
means = json.loads(open(config.DATASET_MEAN).read())
# Initialize several image preprocessors.
# SimplePreprocessor used to resize the images of validation set down to 227*227 pixels.
sp = SimplePreprocessor(227, 227)
# PatchPreprocessor responsible for randomly sample 227*227 regions from the large input images during training time.
pp = PatchPreprocessor(227, 227)
# MeanPreprocessor using for respective, Red, Green, and Blue averages.
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])
# ImageToArrayPreprocessor would handle with convert images to Keras-compatible/Numpy-like arrays.
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators.
trainGen = HDF5DatasetReader(dbPath=config.TRAIN_HDF5,
                             batchSize=config.BATCH_SIZE,
                             preprocessors=[pp, mp, iap],
                             aug=aug,
                             classes=config.NUM_CLASSES)
# Tips: data augmentation is not applied to validation set.
valGen = HDF5DatasetReader(dbPath=config.VAL_HDF5,
                             batchSize=config.BATCH_SIZE,
                             preprocessors=[sp, mp, iap],
                             classes=config.NUM_CLASSES)
# Initialize the optimizer.
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
model.compile(loss="binray_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct the set of callbacks.
plotPath = os.path.sep.join([config.OUTPUT_PATH, "{}".format(os.getpid())])
if not os.path.exists(plotPath):
	os.mkdir(plotPath)

checkpointPath = os.path.sep.join(["{}/{}/".format(config.CHECKPOINT_PATH, os.getpid()), "checkpoint-{epoch:03d}-{val_loss:.4f}.hd5f"])
if not os.path.exists(checkpointPath):
	os.mkdir(plotPath)

checkpoint = ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
trainingMonitor = TrainingMonitor(plotPath)
callbacks = [trainingMonitor, checkpoint]

# Training stage.
print("[INFO] training model...")
# Don't just use aug.flow() as the first position parameter to load data is because of the dataset is too large to store in memory,
#So we should use this method（HDF5DatasetReader()） to load data in batches.
model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // config.BATCH_SIZE,
                    epochs=config.EPOCHS,
                    max_queue_size=config.BATCH_SIZE * 2,
                    callbacks=callbacks,
                    verbose=1)

# Serialize stage.
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# Close the dataset.
trainGen.close()
valGen.close()