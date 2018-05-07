"""
Created on Fri May 3 16:43:00 2018

@author: jercas
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
# Haar Cascade used to detect faces in images or video(optional).
ap.add_argument("-c", "--cascade", required=True, help="path to where face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Lode the face detector cascade and smile detector CNN.
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
# If a video path was not supplied, grab the reference to the webcam.
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# Otherwise, load the video.
else:
	camera = cv2.VideoCapture(args["video"])

# Keep looping until manually stop the script or reach the end of a video.
while True:
	# Grab the current frame.
	(grabbed, frame) = camera.read()
	# If we are viewing a video and we did not grab a frame, then we have reached the end of the video.
	if args.get("video") and not grabbed:
		break
	# Resize the frame, convert it to gray-scale and then clone the original frame so we can draw on it later in the program.
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	# The .detectMultiScale() method handles detecting the bounding box (x,y)-coordinates of faces in the grabbed frame
	#returns a list of 4-tuples that make up the rectangle that bounds the face in the frame (the first two values in this list
	#are the 'starting (x,y)-coordinates' and the last two values are the 'width and height of the bounding box', respectively).

	# Here pass in gray-scale image and indicate that for a given region to be considered a face it must have a minimum
	#width of 30*30 pixels.
	# The 'minNeighbors' attribute helps prune false-positives while the 'scaleFactor' controls the number of image pyramid
	#levels generated.
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	# Loop over the face bounding boxes.
	for (fX, fY, fW, fH) in rects:
		# Extract the ROI of the face from the gray-scale image, resize it to a fixed 28*28 pixels, and then prepare the
		#ROI for classification via the CNN.
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# Determine the probabilities of both "smiling" and "not smiling", then set the label accordingly.
		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"

		# Display the label and bounding box rectangle on the output frame.
		cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
		# Show detected faces along with smiling/not smiling labels.
		cv2.imshow("Face", frameClone)

		# If the 'q' key is pressed, stop the loop.
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

# Cleanup the camera and close any open windows.
camera.release()
cv2.destroyAllWindows()