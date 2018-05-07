"""
Created on Sat Apr 28 17:59:00 2018

@author: jercas
"""
from imutils import paths
from icecream import ic
import argparse
import imutils
import cv2
import os

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images.")
ap.add_argument("-o", "--output", required=True, help="path to output directory of annotations.")
args = vars(ap.parse_args())

# Grab the image paths then initialize the dictionary of character counts.
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# Loop over the image paths.
for (i, imagePath) in enumerate(imagePaths):
	# Display an update to the user
	print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))

	try:
		# Load the image and covert it to gray-scale.
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Padding the image with 8 pixels in every direction to ensure digits caught on the border of the image are retained.
		#(just in case any of digits images are touching the border of the image.
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

		# Threshold the image to reveal the digits(A typical assumption when working with many image processing functions with OpenCV).
		# This function call automatically thresholds image such that is now binary-black pixels represent the 'background'
		#while white pixels are foreground.(In the previous, foreground-white pixels and background-black pixels.)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# Find contours in the image(outlines of each of the digits), keeping only the four largest ones(four digits).
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# In case there is 'noise' in the image so sort the contours by area size, keeping only the four largest one(i.e., digits themselves).
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
		ic(cnts)
		# Given contours we can extract each of them by computing the bounding box((x,y)-coordinates of the digit region).
		# Loop over the contours.
		for c in cnts:
			# Compute the bounding box for contour then extract the digit.
			(x, y, w, h) = cv2.boundingRect(c)
			# Extract the ROI.
			roi = gray[y-5:y+h+5, x-5:x+w+5]

			# Display the character, resizing it larger enough for us to see, then wait for a keypress as the label for the digit.
			cv2.imshow("ROI", imutils.resize(roi, width=28))
			key = cv2.waitKey()
			# if the "‘" key is pressed, then ignore the character.
			if key == ord("‘"):
				print("[INFO] ignoring character")
				continue
			# Grad the key that was pressed and construct the path to the output directory.
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])
			# If the output directory does not exist, create it.
			if not os.path.exists(dirPath):
				os.mkdir(dirPath)

			# Write the labeled character to file.
			count = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
			cv2.imwrite(p, roi)

			# Increment the count for the current key.
			counts[key] = count+1

	# We are trying to control-c out of the script, so break from the loop(still need to press a key for the active
	#window to trigger this).
	except KeyboardInterrupt:
		print("[INFO] Maually leaving script!")
		break

	# An unknown error has occurred for a particular image.
	except:
		print("[INFO] Unknown error, skipping image...")