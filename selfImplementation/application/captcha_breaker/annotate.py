"""
Created on Sat Apr 28 17:59:00 2018

@author: jercas
"""
from imutils import paths
import argparse
import imutils
import cv2
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images.")
ap.add_argument("-o", "--output", required=True, help="path to output directory of annotations.")
args = vars(ap.parse_args())

