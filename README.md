# PeopleDetection - Detector por capas

## Detección de personas
### Input:
- Imagen RGB

### Output:
- Posición x_inicial, x_final, y_inicial, y_final (de la Bounding Box).
- Posición x,y de cada una de las 18 partes del cuerpo.

### Post-procesado:
- Determinar posición respecto a la cámara
- Control de sesiones

### Aproximaciones:
- Las que devuelven un esqueleto de 18 puntos 
    - OpenPose - https://github.com/CMU-Perceptual-Computing-Lab/openpose
    - OpenVino - https://software.intel.com/en-us/openvino-toolkit
    - Keras: con pesos de openPose - https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation
    - Posenet - https://github.com/tensorflow/tfjs-models/tree/master/posenet
    
- Las que devuelven sólo las BoundingBox de la persona
	- Yolo - https://github.com/ndaidong/yolo-person-detect
	- Faster R-CNN Pedestrian detection - https://github.com/ChaoPei/faster-rcnn-pedestrian-detection
	- MobileNet SSD Object Detection - https://github.com/djmv/MobilNet_SSD_opencv


### Evaluación:
- Numero de aciertos en el número de personas por frame
- Formato (por frame):  
```csv
 [id_frame, time_stamp, Bbox(min,max), probabilidad (certeza del dato))]  
```
## Detección de caras
### Input:
- Imagen RGB
    - Preferible, el crop de la BoundingBox extraída en el paso de Detección de Personas

### Output:
- Posición x_inicial, x_final, y_inicial, y_final (de la Bounding Box).
- Posición x,y de cada una de las 68 facial landmarks de la cara (También hay modelos de 5 puntos en vez de 68)

### Post-procesado (trabajo actual):
- Determinar microexpresiones por los facial landmarks

### Aproximaciones
- Mediante OpenCV (Haar) - https://github.com/shantnu/FaceDetect/
- Mediante OpenCV (DNN) - https://github.com/opencv/opencv/tree/master/samples/dnn
- Mediante dlib (HoG) - http://dlib.net/face_detector.py.html
- Mediante dlib (MMOD) - http://dlib.net/cnn_face_detector.py.html
- (Comparación entre los anteriores - https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)
- Otros... 
	- Reconocimiento facial - https://github.com/ageitgey/face_recognition (https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
	- Face clustering - https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/

## Predicción de Género y Edad (etnia)
### Input
- Crop de la BB de la cara extraída en Detección de caras

### Output:
- Edad numérico (¿Rangos?)
- Género numérico (0, 1, probabilidad)

## Predicción de Emociones
### Input
- Crop de la BB de la cara extraída en Detección de caras

### Output:
- Emoción dominante
- Disgust
- Surprise
- Sad
- Angry
- Fear
- Happy
- Contempt
- Neutral

### Aproximaciones
https://github.com/thoughtworksarts/EmoPy
https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition-from-stream.py (http://sefiks.com/2018/01/10/real-time-facial-expression-recognition-on-streaming-data/)
https://github.com/amineHorseman/facial-expression-recognition-using-cnn
https://github.com/ShawDa/facial-expression-recognition

# Resultado final esperado

```json 
{
"bbox_body":{ "coords" : [[x, y], [x, y`]], "image": '' },
"bbox_face":{ "coords" : [[x, y], [x', y']], "image": '' },
"skeleton": [ ],
"facial_landmarks": "",
"emotions": ""

}
```
