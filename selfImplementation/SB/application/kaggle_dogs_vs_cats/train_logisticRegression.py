"""
Created on Tue May 29 15:48:00 2018

@author: jercas
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import argparse
import pickle
import h5py
import os

def main():
    # Construct the argument parse and parse arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--db", default="./dataset/hdf5/features.hdf5", help="path to HDF5 dataset")
    ap.add_argument("-m", "--model", default="./model/dogs_vs_cats.pickle", help="path to output model")
    ap.add_argument("-o", "--output", default="./output", help="path to output plot/report")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyper-parameters")
    args = vars(ap.parse_args())

    DATA_PATH = args["db"]
    OUTPUT_PATH = args["output"]
    MODEL_PATH = args["model"]
    JOBS = args["jobs"]

    # Open the HDF5 database for reading then determine the index of the training and testing split, provided that this data 1
    #was already shuffled *prior* to writing it to disk.
    db = h5py.File(DATA_PATH, "r")
    i = int(db["labels"].shape[0] * 0.75)

    # Perform a grid search over the C hyper-parameter of the LogisticRegression classifier.
    # Define the set of parameters that we want to tune then start a grid search where we evaluate out model for each value of C.
    print("[INFO] tuning hyper-parameters...")
    parameters = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
    model = GridSearchCV(LogisticRegression(), parameters, cv=3, n_jobs=JOBS)
    model.fit(db["features"][:i], db["labels"][:i])
    print("[INFO] best hyper-parameters: {}".format(model.best_params_))

    # Once we’ve found the best choice of C, we can generate a classification report for the testing set.
    preds = model.predict(db["features"][i:])
    report = classification_report(db["labels"][i:],
                                   preds,
                                   target_names=db["label_names"])
    print(report)

    # Compute the raw acc with extra precision.
    acc = accuracy_score(db["labels"][i:], preds)
    print("[INFO] score: {}".format(acc))

    # Serialize report.
    path = [OUTPUT_PATH, "model_classification_report.txt"]
    f = open(os.path.sep.join(path), "w")
    f.write(report)
    f.close()

    # serialize the model to disk.
    print("[INFO] saving model...")
    f = open(MODEL_PATH, "wb")
    f.write(pickle.dumps(model.best_estimator_))
    f.close()

    # close the database.
    db.close()


if __name__ == "__main__":
    main()