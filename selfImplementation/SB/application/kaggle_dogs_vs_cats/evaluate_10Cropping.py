"""
Created on Mon May 28 15:53:00 2018

@author: jercas
"""

from config import dogs_vs_cats_config as config
from SB.preprocessing.MeanPreprocessor import MeanPreprocessor
from SB.preprocessing.CropPreprocessor import CropPreprocessor
from SB.preprocessing.SimplePreprocessor import SimplePreprocessor
from SB.preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SB.io.hdf5DatasetReader import HDF5DatasetReader
from SB.utils.ranked import rank5_acc

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import progressbar
import json

# Load the RGB means for the training set.
means = json.loads(open(config.DATASET_MEAN).read())

# Initialize several image preprocessors.
# SimplePreprocessor used to resize the images of validation set down to 227*227 pixels.
sp = SimplePreprocessor(227, 227)
# CropPreprocessor responsible for oversampling to grab 10 parts of the input image to obtain the average classification acc.
cp = CropPreprocessor(227, 227)
# MeanPreprocessor using for respective, Red, Green, and Blue averages.
mp = MeanPreprocessor(rMean=means["R"], gMean=means["G"], bMean=means["B"])
# ImageToArrayPreprocessor would handle with convert images to Keras-compatible/Numpy-like arrays.
iap = ImageToArrayPreprocessor()

# Loading the pretrained model.
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# Initialize the testing dataset generator, then make predictions on the testing data.
# First obtain a baseline on the testing set using only the original testing images as input to the trained model.
print("[INFO] predicting on test data (no crops)...")
# Preprocessors: use the SimplePreprocessor to resize the 256256 input images down to 227*227 pixels, followed by mean
#normalization and converting the batch to a Keras-compatible/Numpy-like array of images.
testGen = HDF5DatasetReader(config.TRAIN_HDF5, config.BATCH_SIZE/2, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(),
                                      steps=testGen.numImages // (config.BATCH_SIZE/2),
                                      max_queue_size= config.BATCH_SIZE)
# Compute the rank-1 acc.
(rank1, _) = rank5_acc(predictions, testGen.db["labels"])
print("[INFO] rank-1 : {:.2f}".format(rank1 * 100))
testGen.close()


# Re-initialize the testing set generator.
# Then obtain the increased performance on the testing set using the 10-crop testing images as input to the trained model.
print("[INFO] prediction on the test data (with crops)...")
# Preprocessors: this time excluding the 'SimplePreprocessor' and this time instructing it to use just the MeanPreprocessor
#– we’ll apply both over-sampling and Keras-array conversion later in the pipeline.
testGen = HDF5DatasetReader(config.TRAIN_HDF5, config.BATCH_SIZE/2, preprocessors=[mp], classes=config.NUM_CLASSES)
predictions = []

# Initialize the progress bar.
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
proBar  = progressbar.ProgressBar(maxval=testGen.numImages // (config.BATCH_SIZE/2), widgets=widgets)
proBar.start()

# Loop over a single pass of the test data, since we only perform once forward-propagate to obtain classification probabilities in the evaluating stage.
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
	# Loop over each of the individual images.
	for image in images:
		# Apply the crop preprocessor to the image to generate 10 separate crops, then convert them from images to numpy-like arrays.
		"""
			10-crop pre-processor: converts the image into an array of ten 227227 images which these 227227 crops
								were extracted from the original 256256 batch based on the following parts:
								
								 Center of the image
								 Top-left corner
								 Top-right corner
								 Bottom-right corner
								 Bottom-left corner
								 Corresponding horizontal flips
		"""
		crops = cp.preprocess(image)
		crops = np.array([iap.preprocess(crop) for crop in crops], dtype="float32")

		# Make the predictions on the crops and then average them together to obtain the final prediction.
		preds = model.predict(crops)
		# Average the 10-crop predictions to obtain the final probability of the original input image.
		predictions.append(preds.mean(axis=0))
	# Update the progress bar.
	proBar.update()
proBar.finish()

# Compute the rank-1 acc.
(rank1, _) = rank5_acc(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()