"""
Created on Thu May 10 14:06:00 2018

@author: jercas
"""

from SB.utils.ranked import rank5_acc
import argparse
import pickle
import h5py

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", required=True, help="path to load HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to load pre-trained classifier")
args = vars(ap.parse_args())

# Load the pre-trained classifier.
print("[INFO] loading pre-trained classifier...")
model = pickle.loads(open(args["model"], "rb").read())

# Open the hdf5 database for reading.
db = h5py.File(args["database"], "r")
# Determine the index of the training and testing split, provided that the data was already shuffled prior to writing it
#to the disk.
i = int(db["labels"].shape[0] * 0.75)

# Make predictions on the testing set then compute the rank-1 and rank-5 accuracy.
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_acc(preds, db["labels"][i:])
# Display the rank-1 and rank-5 accuracy.
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

# Close the data stream.
db.close()