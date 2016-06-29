import cv2
import numpy as np
import pickle
import sys
import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pyttsx

# load pre-trained classifier
cascPathFace = 'D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
cascPathEyes = 'D:/opencv/sources/data/haarcascades/haarcascade_eye.xml'
cascPathNose = 'D:/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml'
faceCascade = cv2.CascadeClassifier(cascPathFace)
eyesCascade = cv2.CascadeClassifier(cascPathEyes)
noseCascade = cv2.CascadeClassifier(cascPathNose)

# define the center positions of some sample facial objects 
# TO DO: find some reference where these points should be
eyes_means = np.array([[18, 26],
					   [45, 26]])

nose_means = np.array([31, 39])

# define some allowed distance to the sample facial objects' ideal position
eyes_stds = np.array([4,2])

nose_stds = np.array([3,5])

# define the ideal positions, will be used to define affine transformation
ideal_pts = np.float32(np.vstack([eyes_means, nose_means]))

# load the features of the training examples, 
# the label decoder
# the tuned feature extractor
# the tuned knn model
with open('trained_model.pkl', 'rb') as f:
	(features, le, extractor, model) = pickle.load(f)

# set up tts engine
engine = pyttsx.init()

# make the speech speed a little bit slower, default is a bit too fast
engine.setProperty('rate', engine.getProperty('rate') - 100)

# start capture iamges
video_capture = cv2.VideoCapture(0)

# get the center location for each facial objects
def get_object_centers(facial_objects):
	return np.array([[x + w/2, y + h/2] for (x,y,w,h) in facial_objects])

# calculate euclidean distance between two points
def euclidean_distance(p1,p2):
	return np.sqrt(np.sum((p1-p2)**2))

# distance to target
def get_distance_to_target(facial_objects, target_pos):
	return np.array([euclidean_distance(facial_object, target_pos)
					 for facial_object in facial_objects])

# test and return the most possible valid object
def valid_object_exist(facial_objects, target_pos, allowed_errors):
	distances = get_distance_to_target(
		facial_objects,
		target_pos)
	most_possible_object = facial_objects[
		np.where(distances == np.min(distances))[0]]
	valid = np.sum(np.abs(most_possible_object - target_pos) <= allowed_errors) == 2
	return (most_possible_object, valid)

while True:
	# get the frame
	ret, frame = video_capture.read()

	# convert frame to gray image
	gray = cv2.cvtColor(
		frame,
		cv2.COLOR_BGR2GRAY)

	## TO DO: try to create a object and make this code cleaner
	# detect faces in the gray image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (32, 32),
		flags = cv2.CASCADE_SCALE_IMAGE
		)

	# try to filter the faces
	for (x,y,w,h) in faces:
		# get a face
		face = gray[y:y+h, x:x+w]
		# resize to 64 by 64
		face = cv2.resize(face, (64,64))

		# detect eyes and noses in the face
		eyes = eyesCascade.detectMultiScale(
			face,
			scaleFactor = 1.05
			)

		noses = noseCascade.detectMultiScale(
			face,
			scaleFactor = 1.02
			)

		# get center locations of eyes and noses
		eyes_centers = get_object_centers(eyes)
		noses_centers = get_object_centers(noses)
	
		# test if the face is valid or not
		if (eyes_centers.shape[0] >= 2) & (noses_centers.shape[0] >= 1):
			most_possible_left_eye, valid_left_eye = valid_object_exist(
				eyes_centers,
				eyes_means[0],
				eyes_stds*1.96)

			most_possible_right_eye, valid_right_eye = valid_object_exist(
				eyes_centers,
				eyes_means[1],
				eyes_stds*1.96)

			most_possible_nose, valid_nose = valid_object_exist(
				noses_centers,
				nose_means,
				nose_stds*1.96)

			# if the facial objects are all valid transfer it and try save / identify the person
			if valid_nose & valid_right_eye & valid_left_eye:
				current_pts = np.float32(np.vstack(
					(most_possible_left_eye,
					 most_possible_right_eye,
					 most_possible_nose)))

				# set up the affine transformation
				affine_transformation = cv2.getAffineTransform(current_pts, ideal_pts)

				# transform the face
				transformed_face = cv2.warpAffine(
					face,
					affine_transformation,
					(64,64)
					)

				# if the program is ran with an extra parameter specify the name of the person 
				# then save some images of the person
				if len(sys.argv) > 1:
					f = 'img/' + str(sys.argv[1]) + '_' + str(int(time.time())) + '.png'
					cv2.imwrite(f, transformed_face)

				# extract features from the transformed face image using the tuned extractor loaded				
				feature = extractor.describe(transformed_face)

				# use the tuned model to predict the person
				predicted_name = le.inverse_transform(model.predict(feature)).tolist()[0]

				# print out the predicted name
				print predicted_name

				# speak out a greeting
				engine.say("Hello " + str(predicted_name))
				engine.runAndWait()

	cv2.imshow("cam", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# release resources
video_capture.release()
engine.stop()
cv2.destroyAllWindows()




