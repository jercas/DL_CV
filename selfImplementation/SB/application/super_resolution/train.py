"""
Created on Tue Jun 19 13:54:00 2018

@author: jercas
"""
import matplotlib
matplotlib.use("Agg")

from super_resolution.config import super_resolution_config as config
from SB.io.hdf5DatasetReader import HDF5DatasetReader
from SB.nn.conv.srcnn import SRCNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def super_res_generator(inputDataGen, outputDataGen):
	# Start an infinite loop for the training data.
	while True:
		# Grab the next input images and target outputs, discarding the class labels (which are irrelevant in srcnn).
		# next (a built-in Python function to return the next item in a generator).
		inputData  = next(inputDataGen)[0]
		outputData = next(outputDataGen)[0]
		# Yield a tuple of the input data and output data.
		yield(inputData, outputData)


# Initialize the input images and target output images generators.
inputs  = HDF5DatasetReader(config.INPUTS_DB, config.BATCH_SIZE)
outputs = HDF5DatasetReader(config.OUTPUTS_DB, config.BATCH_SIZE)

# Initialize the model and optimizer.
print("[INFO] compiling model...")
opt = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=3)
# Using mean-squared loss (MSE) rather than binary/categorical cross-entropy and do not need any metrics.
model.compile(loss="mse", optimizer=opt)

# Train model.
print("[INFO] training model...")
# Using super_res_generator to jointly yield training batches from both the inputs and targets generators, respectively.
H = model.fit_generator(super_res_generator(inputs.generator(), outputs.generator()),
                        steps_per_epoch=inputs.numImages // config.BATCH_SIZE,
                        epochs=config.NUM_EPOCHS, verbose=1)

# Serialized the trained model.
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="loss")
plt.title("Loss on super resolution training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

# Close the HDF5 dataset.
inputs.close()
outputs.close()