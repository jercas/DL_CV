"""
Created on Mon Apr 24 16:42:00 2018

@author: jercas
"""

# Logs loss and acc to disk.
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
	def __init__(self, figurePath, jsonPath=None, startAt=0):
		"""
			Constructor function.

			Parameters:
				figurePath: The path to the output plot that use to visualize loss and acc over time.
				jsonPath:  Optional. The path used to serialize the loss and acc values as a JSON file and then can be
									 used to plot the training history.
				startAt: Optional. The starting epoch that training is 'resumed' at when using ctrl+C training.
		"""

		# Store the output path for the figure, the path to the JSON serialized file and the starting epoch.
		super(TrainingMonitor, self).__init__()
		self.figurePath = figurePath
		self.jsonPath = jsonPath
		self.startAt = startAt


	def on_train_begin(self, logs={}):
		"""
			Built-in function of callbacks.Callback class (superclass of BaseLogger) that as the name suggests, is called
			'once' when the training process starts.
		"""
		# Initialize the history dictionary.
		# Contains four keys: train_loss, train_acc, val_loss, val_acc.
		self.H = {}

		# Check to see if the JSON path was supplied and if the JSON history path exists means we need to load existed
		#training history.
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				#  If so, then load the JSON's contents and update the history dictionary H up until the starting epoch.
				#(since that is where we will resume training restart from).
				self.H = json.loads(open(self.jsonPath).read())

				# Check to see if a starting epoch was supplied.
				if self.startAt > 0:
					# Loop over the entries in the history log and trim any entries that are past the starting epoch.
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]


	def on_epoch_end(self, epoch, logs={}):
		"""
			Built-in function of callbacks.Callback class (superclass of BaseLogger) that as the name suggests, is called
			at the end of every epochs and automatically supplied to parameters 'epoch' from keras on the background.

			Parameter:
				epoch: an integer representing the epoch number.
				logs: a dictionary which contains the training and validation loss+acc for current epoch.
		"""
		# Loop over the logs and update the loss, acc, etc. for the entire training process.
		for (k, v) in logs.items():
			# The k and v is a certain key and its value in current epoch(automatically feed by keras).
			l = self.H.get(k, [])
			# Maintain a list of values for each of four keys above, each list is updated at the end of every epoch.
			# Thus, enabling to plot an updated loss and acc curve as soon as a epoch completes instead of only can plot
			#the curves at the end of the whole training phase end in the previous.
			l.append(v)
			self.H[k] = l

		# Check to see if the training history should be serialized to file.
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		# Ensure at least two epochs have passed before plotting (epoch starts at zero).
		if len(self.H["loss"]) > 1:
			# Plot the training loss and acc.
			xAxis = np.arange(0, len(self.H["loss"]))
			# Plot phase.
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(xAxis, self.H["loss"], label="train_loss")
			plt.plot(xAxis, self.H["val_loss"], label="val_loss")
			plt.plot(xAxis, self.H["acc"], label="train_acc")
			plt.plot(xAxis, self.H["val_acc"], label="val_acc")
			plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()
			plt.savefig(self.figurePath)
			print("[INFO] {} has saved...".format(self.figurePath))
			plt.close()

