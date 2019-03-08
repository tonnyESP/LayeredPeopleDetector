#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
import os
import sys
import time
from PIL import Image

#from imutils import face_utils
import dlib

import base64

from flask import Flask, jsonify, request, redirect, render_template

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_url_path='/static', template_folder='/templates')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/', methods=['GET'])
def display_result():
    return render_template('placeholder_faces.html')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            
            filename = file.filename # save file 
            filepath = os.path.join('', filename)
            file.save(filepath)
            image = cv2.imread(filepath)            

            
            
            to_return = person_detector(image)
            
            return jsonify()

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>People feature extraction</title>
    <h1>Upload a picture and return people in it!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''



def facecrop_opencv(img):
  global cascade

  minisize = (img.shape[1], img.shape[0])
  miniframe = cv2.resize(img, minisize)

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
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # detect faces in the grayscale image
  rects = detector(gray, 0)
  print(str(rects))
  print(str(len(rects)))
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
      retval, bufferval = cv2.imencode('.jpg', sub_face)
      jpg_as_text = base64.b64encode(bufferval).decode('utf-8')

      result = {
          "b64_person": str(jpg_as_text),
          "x": str(x),
          "y": str(y),
          "w": str(w),
          "h": str(h)
      }

      return result;

  return None;

def person_detector(img):
    img_resized = cv2.resize(img,(300,300)) # resize img for prediction
    heightFactor = img.shape[0]/300.0
    widthFactor = img.shape[1]/300.0 

    tic = time.time()


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

    toc = time.time()
    print ("MobileSSD time : "+ str(toc-tic) + " seconds")

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


                tic = time.time()

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
                retval, bufferval = cv2.imencode('.jpg', person_bb)
                jpg_as_text = base64.b64encode(bufferval).decode('utf-8')

                #cv2.imwrite('results/person_'+str(i+1)+'.jpg', person_bb)

                detected_person["bbox_body"] = {
                    "b64_person": str(jpg_as_text),
                    "x": str(xLeftBottom_),
                    "y": str(yLeftBottom_),
                    "w": str(xRightTop_ - xLeftBottom_),
                    "h": str(yRightTop_ - yLeftBottom_)
                }

                toc = time.time()
                print ("\t Crop person "+str(i)+" : "+ str(toc-tic) + " seconds")

                # detect faces in the grayscale image
                tic = time.time()
                person_face_bb_dlib = facecrop_dlib(person_bb.copy())

                toc = time.time()
                print ("\t\t Crop face person "+str(i)+" : "+ str(toc-tic) + " seconds")

                if person_face_bb_dlib is not None:
                  detected_person["bboxface_dlib"] = person_face_bb_dlib;
                  #with open('results/person_'+str(i+1)+'_face_dlib.jpg', "wb") as fh:
                  #  fh.write(base64.decodestring(person_face_bb_dlib["b64_person"]))

                tic = time.time()

                person_face_bb_opcv = facecrop_opencv(person_bb.copy())

                toc = time.time()
                print ("\t\t Crop face person "+str(i)+" : "+ str(toc-tic) + " seconds")

                if person_face_bb_opcv is not None:
                  detected_person["bboxface_opcv"] = person_face_bb_opcv
                  #with open('results/person_'+str(i+1)+'_face_opencv.jpg', "wb") as fh:
                  #  fh.write(base64.decodestring(person_face_bb_opcv["b64_person"]))
                
                result.append(detected_person);

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


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=5001, debug=True)