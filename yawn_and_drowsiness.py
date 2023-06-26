'''
    pyimagesearch
    Drowsiness detection with OpenCV
    by Adrian Rosebrock, 2017
    https://pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
'''

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pygame

def play_audio(file_path):
	pygame.mixer.init()
	pygame.mixer.music.load(file_path)
	pygame.mixer.music.play()

def cal_yawn(shape): 
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))
	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))
	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)
	distance = dist.euclidean(top_mean,low_mean)
	
	return distance

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)
	
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	
    # return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 48

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
YAWN_ALARM_ON = False
DROWSINESS_ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	
    # loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
	
        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		lip = shape[48:60]
		cv2.drawContours(frame,[lip],-1,(0, 165, 255),thickness=3)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		lip_dist = cal_yawn(shape)
		if lip_dist > 22 : 
			if not YAWN_ALARM_ON:
				YAWN_ALARM_ON = True
					
				# check to see if an alarm file was supplied,
				# and if so, start a thread to have the alarm
				# sound played in the background
				if args["alarm"] != "":
					t = Thread(target=play_audio, args=(args["alarm"],))
					t.start()

			cv2.putText(frame, f'YAWN ALERT!',(10,300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
		
		else:
			YAWN_ALARM_ON = False
		
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not DROWSINESS_ALARM_ON:
					DROWSINESS_ALARM_ON = True
					
					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=play_audio, args=(args["alarm"],))
						t.start()
						
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			DROWSINESS_ALARM_ON = False
		
        # draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "LIP: {:.2f}".format(lip_dist), (300, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
	
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
