"""
Created on Sun Apr 11 12:50:00 2018

@author: jercas
"""
from skimage.exposure import rescale_intensity
import numpy as np
import icecream as ic
import argparse
import cv2

def convolve(image, K):
	# Grab the spatial dimensions of the image(grayscale) and kernel.
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]

	# Allocate memory for the output image, taking care to "pad" the borders of the input image so spatial size(i.e.,
	#width and height) are not reduced.
	pad = (kW - 1) // 2

	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float")

	# Loop over the input image, "sliding" the kernel across each (x,y)-coordinate from left-to-right and top-to-bottom,
	#one pixel at a time.
	for y in np.arange(pad, pad+iH):
		for x in np.arange(pad, pad+iW):
			# Extract the ROI(Region of Interest which will be centered around the current(x,y)-coordinates of the image
			#and have the same size as kernel) of the image by extracting the "center" region of the current (x,y)-
			#coordinates dimensions.
			roi = image[y-pad : y+pad+1, x-pad : x+pad+1]

			# Perform the actual convolution by taking the element-wise multiplication between ROI and the kernel,
			#then sum up the matrix as a single value.
			kernelOutput = (roi * K).sum()

			# Store the convolved value in the output (x,y)-coordinate of the output image.
			output[y-pad, x-pad] = kernelOutput
	# Rescale the output image to be in the range [0, 255], because of the value in output image would fall outside this
	#range after convolution .
	output = rescale_intensity(output, in_range=(0, 255))
	# Convert image back to an unsigned 8-bit integer data type.
	# Previously, the output image was a floating point type in order to handle pixel value outside the range.
	output = (output * 255).astype("uint8")

	return output


# DRIVER PORTION:
# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# *Following kernels are all "manually bulit" to perform a given operation*

# Construct "average blurring" kernels used to "smooth" the image.
# Why this kernel perform Blurring?
# Because each entry in the kernel is an average of 1/N where N is the total number of entries in the matrix.
# Thus, the kernel will multiply each input pixel by a small fraction and take the sum -- exactly the definition of the average.
smallBlur  = np.ones((7,7),   dtype="float") * (1.0 / (7*7))
middleBlur = np.ones((11,11), dtype="float") * (1.0 / (11*11))
largeBlur  = np.ones((21,21), dtype="float") * (1.0 / (21*21))

# Construct a "sharpening" kernel/filter.
sharpen = np.array((
	[0,  -1,  0],
	[-1,  5, -1],
	[0,  -1,  0]), dtype="int")

# Construct a "Laplacian" kernel used to "detect the edge-like region" in the image.
laplacian = np.array((
	[0,  1, 0],
	[1, -4, 1],
	[0,  1, 0]), dtype="int")

# Construct a "Sobel" x-axis kernel used to detect edge-like regions along the x-axis.
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

# Construct a "Sobel" y-axis kernel used to detect edge-like regions along the y-axis.
sobelY = np.array((
	[-1, -2, -1],
	[0,  0,  0],
	[1,  2,  1]), dtype="int")

# Construct an "emboss" kernel.
emboss = np.array((
	[-2, -1, 0],
	[-1, 1, 1],
	[0, 1, 2]), dtype="int")

kernelBank = (
	("small_blur", smallBlur),
	("middle_blur", middleBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY),
	("emboss", emboss)
)

# Load the input image and covert it to grayscale.
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Loop over all the kernels.
for (kernelName, K) in kernelBank:
	# Apply the kernel to the grayscale image using both custom "convolve" function and OpenCV's "filter2D' function.
	print("[INFO] applying {} kernel...".format(kernelName))
	convolveOutput = convolve(gray, K)
	# cv2.filter2D() is OpenCV's much more optimized version of convolve function in this script.
	# Using this function here is to sanity check and compare custom implementation with library implementation.
	opencvOutput = cv2.filter2D(gray, -1, K)

	# Show the output image.
	cv2.imshow("Original", image)
	cv2.imshow("Gray", gray)
	cv2.imwrite('./output/gray.png', gray)
	cv2.imshow("{} - convolve: ".format(kernelName), convolveOutput)
	#cv2.imwrite('./output/{}-convolve.png'.format(kernelName), convolveOutput)
	cv2.imshow("{} - opencv: ".format(kernelName), opencvOutput)
	#cv2.imwrite('./output/{}-opencv.png'.format(kernelName), opencvOutput)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

