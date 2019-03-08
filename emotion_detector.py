import numpy as np
import cv2
from keras.preprocessing import image
import tensorflow as tf
import posenet
import argparse
import os
import sys
import time
import datetime
from PIL import Image
import random
from pymongo import MongoClient

from colors import get_colors

#from imutils import face_utils
import dlib

import base64


def facecrop_opencv(img):
  global cascade

  minisize = (img.shape[1], img.shape[0])
  try:
  	miniframe = cv2.resize(img, minisize)
  except:
  	miniframe = None
  	
  all_faces = cascade.detectMultiScale(miniframe)
  if len(all_faces) > 0:
	  # Only get the first face detected
	  x, y, w, h = [ v for v in all_faces[0] ]
	  #cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

	  sub_face = img[y:y+h, x:x+w]
	  retval, bufferval = cv2.imencode('.jpg', sub_face)
	  jpg_as_text = base64.b64encode(bufferval).decode('utf-8')

	  result = {
		  "b64_face": str(jpg_as_text),
		  "x": str(x),
		  "y": str(y),
		  "w": str(w),
		  "h": str(h)
	  }
	  return result

  return None;

def facecrop_dlib(img):
  global detector, predictor
  
  try:
	  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	  # detect faces in the grayscale image
	  rects = detector(gray, 0)
	  #print(str(rects))
	  #print(str(len(rects)))
  except:
	  rects = []
	  pass
  # loop over the face detections
  if len(rects) > 0:
	  rect = rects[0]
	  # determine the facial landmarks for the face region, then
	  # convert the facial landmark (x, y)-coordinates to a NumPy
	  # array
	  '''shape = predictor(gray, rects[0])
	  shape = face_utils.shape_to_np(shape)

	  # loop over the (x, y)-coordinates for the facial landmarks
	  # and draw them on the image
	  for (x, y) in shape:
		  cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
	  '''
	  x = rect.tl_corner().x
	  y = rect.tl_corner().y

	  w = rect.width()
	  h = rect.height()

	  sub_face = img[y:y+h, x:x+w]
	  
	  try:
		  retval, bufferval = cv2.imencode('.jpg', sub_face)
		  jpg_as_text = base64.b64encode(bufferval).decode('utf-8')
	  except:
		  jpg_as_text = ""
		  
	  result = {
		  "b64_person": str(jpg_as_text),
		  "x": str(x),
		  "y": str(y),
		  "w": str(w),
		  "h": str(h)
	  }

	  return result

  return None

def person_detector(img):
	img_resized = cv2.resize(img,(300,300)) # resize img for prediction
	heightFactor = img.shape[0]/300.0
	widthFactor = img.shape[1]/300.0 

	#tic = time.time()


	# MobileNet requires fixed dimensions for input image(s)
	# so we have to ensure that it is resized to 300x300 pixels.
	# set a scale factor to image because network the objects has differents size. 
	# We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
	# after executing this command our "blob" now has the shape:
	# (1, 3, 300, 300)
	blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
	#Set to network the input blob 
	net.setInput(blob)
	
	#Prediction of network
	detections = net.forward()

	#toc = time.time()
	#print ("MobileSSD time : "+ str(toc-tic) + " seconds")

	#Size of img resize (300x300)
	cols = img_resized.shape[1] 
	rows = img_resized.shape[0]

	result = []

	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2] #Confidence of prediction 
		if confidence > thr: # Filter prediction 
			class_id = int(detections[0, 0, i, 1]) # Class label

			# Draw label and confidence of prediction in frame resized
			#if class_id in [classNames]:
			if class_id == 15: # Only detect people


				#tic = time.time()

				detected_person = {}

				# Object location 
				xLeftBottom = int(detections[0, 0, i, 3] * cols) 
				yLeftBottom = int(detections[0, 0, i, 4] * rows)
				xRightTop   = int(detections[0, 0, i, 5] * cols)
				yRightTop   = int(detections[0, 0, i, 6] * rows)

				xLeftBottom_ = int(widthFactor * xLeftBottom) 
				yLeftBottom_ = int(heightFactor* yLeftBottom)
				xRightTop_   = int(widthFactor * xRightTop)
				yRightTop_   = int(heightFactor * yRightTop)

				# Crop the bounding box for the person
				# we need the .copy() to obtain new bb not overwriting the img
				# also, note x y are flipped

				person_bb = img[yLeftBottom_:yLeftBottom_+(yRightTop_-yLeftBottom_), xLeftBottom_:xLeftBottom_+(xRightTop_-xLeftBottom_)].copy()
				
				try:
					retval, bufferval = cv2.imencode('.jpg', person_bb)
					jpg_as_text = base64.b64encode(bufferval).decode('utf-8')
				except:
					jpg_as_text = ""
				#cv2.imwrite('results/person_'+str(i+1)+'.jpg', person_bb)

				detected_person["bbox_body"] = {
					"b64_person": str(jpg_as_text),
					"x": str(xLeftBottom_),
					"y": str(yLeftBottom_),
					"w": str(xRightTop_ - xLeftBottom_),
					"h": str(yRightTop_ - yLeftBottom_)
				}

				#toc = time.time()
				#print ("\t Crop person "+str(i)+" : "+ str(toc-tic) + " seconds")

				# detect faces in the grayscale image
				#tic = time.time()
				person_face_bb_dlib = facecrop_dlib(person_bb.copy())

				#toc = time.time()
				#print ("\t\t Crop face person "+str(i)+" : "+ str(toc-tic) + " seconds")

				if person_face_bb_dlib is not None:
				  detected_person["bboxface_dlib"] = person_face_bb_dlib
				  #with open('results/person_'+str(i+1)+'_face_dlib.jpg', "wb") as fh:
				  #  fh.write(base64.decodestring(person_face_bb_dlib["b64_person"]))
				
				#tic = time.time()

				person_face_bb_opcv = facecrop_opencv(person_bb.copy())

				#toc = time.time()
				#print ("\t\t Crop face person "+str(i)+" : "+ str(toc-tic) + " seconds")

				if person_face_bb_opcv is not None:
				  detected_person["bboxface_opcv"] = person_face_bb_opcv
				  #with open('results/person_'+str(i+1)+'_face_opencv.jpg', "wb") as fh:
				  #  fh.write(base64.decodestring(person_face_bb_opcv["b64_person"]))
				
				result.append(detected_person)

	cv2.imwrite("results/frame.png", img)
	return result
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# Initialize OPENCV face detector
facedata = "models/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(facedata)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Initialize MobileNetSSD object detector net
prototxt = 'models/MobileNetSSD/MobileNetSSD_deploy.prototxt'
weights = 'models/MobileNetSSD/MobileNetSSD_deploy.caffemodel'
thr = 0.4

# Labels of Network
classNames = { 0: 'background',
	1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
	5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
	10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
	14: 'motorbike', 15: 'person', 16: 'pottedplant',
	17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(prototxt, weights)



#-----------------------------
#opencv initialization

source = '/home/tonny/keras_Realtime_Multi-Person_Pose_Estimation/output.mp4'
cap = cv2.VideoCapture(source)
cap.set(3, 640)
cap.set(4, 480)
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("models/facial_expression/facial_expression_model_structure.json", "r").read())
model.load_weights('models/facial_expression/facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('enfado', 'disgusto', 'miedo', 'felicidad', 'tristeza', 'sorpresa', 'neutral')

colors = get_colors()

client = MongoClient('localhost', 12334)

db = client['people-detection']
collection = db['raw-data']

with tf.Session() as sess:
	model_cfg, model_outputs = posenet.load_model(101, sess)
	output_stride = model_cfg['output_stride']

	frame_counter = 0
	while( True or frame_counter > 500):
		if (frame_counter % 3):
			frame_counter += 1
			continue

		ret, img = cap.read()
		people_detected = person_detector(img)
		counter = 0
		for person in people_detected:

			to_return = {}
			to_return["frame"] = frame_counter
			to_return["datetime"] = datetime.datetime.utcnow()
			to_return["source"] = source
			to_return["person"] = person
			
			color = colors[counter]
			counter += 1
			# 1 rect person bbox
			x = int(person["bbox_body"]["x"])
			y = int(person["bbox_body"]["y"])
			w = int(person["bbox_body"]["w"])
			h = int(person["bbox_body"]["h"])
			cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
			
			try:
				input_image, display_image, output_scale = posenet.read_img(
					img[y:y+h, x:x+w], scale_factor=0.7125, output_stride=output_stride)

				heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
					model_outputs,
					feed_dict={'image:0': input_image}
				)

				pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
					heatmaps_result.squeeze(axis=0),
					offsets_result.squeeze(axis=0),
					displacement_fwd_result.squeeze(axis=0),
					displacement_bwd_result.squeeze(axis=0),
					output_stride=output_stride,
					max_pose_detections=1,
					min_pose_score=0.15)

				keypoint_coords *= output_scale

				to_return["person"]["pose"] = {}
				to_return["person"]["pose"]["pose_scores"] = dict(pose_scores)
				to_return["person"]["pose"]["keypoint_scores"] = dict(keypoint_scores)
				to_return["person"]["pose"]["keypoint_coords"] =  dict(keypoint_coords)


				# TODO this isn't particularly fast, use GL for drawing and display someday...
				'''overlay_image = posenet.draw_skel_and_kp(
					display_image, pose_scores, keypoint_scores, keypoint_coords,
					min_pose_score=0.15, min_part_score=0.1)
				'''
				
			except:
				pass

			# If face is detected, show bbox
			try:
				x = int(person["bbox_body"]["x"]) + int(person["bboxface_opcv"]["x"])
				y = int(person["bbox_body"]["y"]) + int(person["bboxface_opcv"]["y"])
				w = int(person["bboxface_opcv"]["w"])
				h = int(person["bboxface_opcv"]["h"])
				cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
				detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
				detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				
				img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
				
				predictions = model.predict(img_pixels) #store probabilities of 7 expressions

				#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
				max_index = np.argmax(predictions[0])
				
				emotion = emotions[max_index]
				
				to_return["person"]["emotions"] = {}
				to_return["person"]["emotions"]["predictions"] = predictions
				to_return["person"]["emotions"]["max"] = max_index				

				#write emotion text above rectangle
				cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			except:
				pass
		
			try:
				#print (to_return)
				collection.insert(to_return)
				print("guardado frame"+str(frame_counter))
			except Exception as e:
				print("ERROR EN FRAME "+ str(frame_counter) + " " + str(e) )
			cv2.imshow('img',img)
		frame_counter += 1
		#if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		#	break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()