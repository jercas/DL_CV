"""
Created on Thu May 10 09:37:00 2018

@author: jercas
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import argparse
# pickle: used to serialize LogisticRegression model to disk after training.
import pickle
import h5py

def main():
	# Construct the argument parse and parse the arguments.
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--database", required=True, help="path HDF5 database")
	ap.add_argument("-m", "--model", required=True, help="path to output model")
	ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyperparameters")
	args = vars(ap.parse_args())

	# Open the HDF5 database for reading then determine the index of the training and testing split, provided that this data
	#was 'already shuffled' prior to writing it to disk.
	db = h5py.File(args["database"], "r")
	# The boundary of training set(~[:i]) and test set(~[i:]).
	i = int(db["labels"].shape[0] * 0.75)

	# Define the set of parameters that we want to tune then start a grid search where we evaluate out model for each value of C.
	print("[INFO] tuning hyperparameters...")
	params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
	model  = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
	model.fit(db["features"][:i], db["labels"][:i])

	# Once the best hyperparameters be founded.
	print("[INFO] best hyperparameter: {}".format(model.best_params_))

	# Evaluate the model.
	print("[INFO] evaluating...")
	preds = model.predict(db["features"][i:])
	print(classification_report(db["labels"][i:], preds, target_names=[lb_name.split('/')[-1] for lb_name in db["label_names"]]))

	# Serialize the model to disk.
	print("[INFO] saving model...")
	f = open(args["model"], "wb")
	f.write(pickle.dumps(model.best_estimator_))

	# End buffer stream.q
	f.close()
	db.close()


if __name__ == "__main__":
	main()