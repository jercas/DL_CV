"""
Created on Thu Jun 14 10:19:00 2018

@author: jercas
"""

from SB.nn.conv.dcgan import DCGAN
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

# Construct the argument parser and parse the argument.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="./output/dcgan_mnist", help="path to the output dictionary")
ap.add_argument("-e", "--epochs", type=int, default=50, help="# epoch to train for")
ap.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for training")
args = vars(ap.parse_args())

OUTPUT = args["output"]
# Store the hyper-parameters in convince variables.
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

# Load the MNIST dataset and stack the training and testing data points to obtain additional train data.
print("[INFO] loading MNIST dataset...")
# Ignoring the class labels here since we do not need them — just only interested in the actual pixel data.
"""
	There is not a concept of a “test set” for GANs. Our goal when training isn’t minimal loss or high accuracy. 
	Instead, we are seeking an equilibrium between the generator and the discriminator.
	To help us obtain this equilibrium we can combine both the training and testing images to give us additional training data.
"""
((trainX, _), (testX, _)) = mnist.load_data()
trainImages = np.concatenate([trainX, testX])

# Add in an extra dimension for the channel and scale the images into the range [-1, 1] (which is the output range of the tanh function).
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5

# Build the generator.
print("[INFO] building the generator...")
# The generator that will transform the input random vector to a volume of shape 7x7x64-channel map.
gen = DCGAN.build_generator(dim=7, depth=64, channels=1)
# Build the discriminator.
print("[INFO] building the discriminator...")
disc = DCGAN.build_discriminator(width=28, height=28, depth=1)
discOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
# Using binary cross-entropy loss here as (1) our discriminator has a sigmoid activation function that will (2) return a
#probability indicating whether the input image is real vs. fake.
# Since there are only two class labels (real and synthetic) we use binary cross-entropy loss.
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

# Build the adversarial model by first setting the discriminator to *not be trainable*, then compile the generator and
#discriminator together.
print("[INFO] building GAN...")
# Freeze the discriminator weights.
disc.trainable = False
# The input to the GAN will be a random vector that is 100-dimensions.
ganInput  = Input(shape=(100,))
# Pass the random vector through the generator first and then the output will go to the discriminator next.
# As the discriminator weights are frozen at this point so the feedback from the discriminator will enable the generator
#to learn how to generate better synthetic images.
ganOutput = disc(gen(ganInput))
# Combine two models.
gan = Model(ganInput, ganOutput)
# Default: setting the learning rate of the actual GAN to be half of the discriminator is a good starting point.
# While this process worked for this experiment's architecture + dataset.
ganOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=ganOpt)

# Randomly generate some benchmark noise so we can consistently visualize how the generative modeling is learned.
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))
# Loop over the epochs.
for epoch in range(NUM_EPOCHS):
	# Show epoch information and compute the number of batches per epoch.
	print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
	batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

	# Loop over the batches.
	for i in range(0, batchesPerEpoch):
		# Initialize an (empty) output path.
		path = None

		# Select the next batch of images, then randomly generate noise for the generator to predict on.
		imageBatch = trainImages[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
		# Generate images using the noise + generator model.
		genImages = gen.predict(noise, verbose=0)

		# Concatenate the *actual* images and the *generator* images, construct class labels for the discriminator and shuffle the data.
		X = np.concatenate((imageBatch, genImages))
		# Each real image will have a class label of 1 while every fake image will be labeled 0.
		y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
		(X, y) = shuffle(X, y)
		# Train the discriminator on the data.
		discLoss = disc.train_on_batch(x=X, y=y)

		# Train generator via the adversarial model by (1) generating random noise and (2) training the generator with
		#the discriminator weights frozen.
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
		"""
			We first generate a total of BATCH_SIZE random vectors. However, unlike in our previous code block where we 
		were nice enough to tell our discriminator what is real vs. fake, we’re now going to try to trick the discriminator 
		by labeling the random noise as a real image.
			The feedback from the discriminator enables use to actually train the generator (keeping in mind that the 
		discriminator weights are frozen for this operation). Not only is looking at the loss values important when training 
		a GAN, but you also need to visually examine the output of the gan on your benchmarkNoise.
		"""
		ganLoss = gan.train_on_batch(x=noise, y=[1] * BATCH_SIZE)

		# Check to see if this is the end of an epoch, and if so, initialize the output path.
		if i == batchesPerEpoch - 1:
			path = [OUTPUT, "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]
		# Otherwise, check to see if should visualize the current batch for the epoch.
		else:
			# Create more visualizations early in the training process.
			if epoch < 10 and i % 25 == 0:
				path = [OUTPUT, "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

			# Visualizations later in the training process are less interesting.
			elif epoch >= 10 and i % 100 == 0:
				path = [OUTPUT, "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

		# Check to see if we should visualize the output of the generator model on our benchmark data.
		if path is not None:
			# Show loss information.
			print("[INFO] Step {}_{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}".format(epoch + 1, i, discLoss, ganLoss))
			# Make predictions on the benchmark noise, scale it back to the range [0, 255] and generate the montage.
			images = gen.predict(benchmarkNoise)
			images = ((images * 127.5) + 127.5).astype("uint8")
			# Since we are generating single channel images, we repeat the grayscale channel three times to construct a
			#3-channel RGB image.
			images = np.repeat(images, 3, axis=-1)
			visualization = build_montages(images, (28, 28), (16, 16))[0]

			# Serialize the visualization.
			path = os.path.sep.join(path)
			cv2.imwrite(path, visualization)