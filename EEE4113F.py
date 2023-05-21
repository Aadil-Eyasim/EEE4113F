# Import packages
import os
import cv2
import numpy as np
import importlib.util
import time
from datetime import datetime
import urllib.request
import requests

# Define ESP32-CAM IP and URLs
esp32cam_ip = "192.168.137.69"  # Replace with the IP address of your ESP32-CAM
temperature_url = f"http://{esp32cam_ip}/temperature"
humidity_url = f"http://{esp32cam_ip}/humidity"

url='http://192.168.137.69/capture'
im=None

GRAPH_NAME = '/home/aadil/objdetec/Sample_TFLite_model/detect.tflite'
LABELMAP_NAME = '/home/aadil/objdetec/Sample_TFLite_model/labelmap.txt'
output_folder = 'PERSON'
output_folder1 = 'PHONE'
output_folder2 = 'CAT'

# Import TensorFlow libraries
# Import interpreter from tflite_runtime
pkg = importlib.util.find_spec('tflite_runtime')
from tflite_runtime.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

#parameters to calculate frame per second    
counter = 0
fps = 0
start_time = time.time()
fps_avg_frame_count = 10

# Initialise a variable to store the current frame
frame = None

# Define a function to download and decode a video frame from the camera
def get_frame():
    global frame
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    frame = cv2.imdecode(arr, -1)

# Function to get temperature from ESP32-CAM
def get_temperature():
    response = requests.get(temperature_url)
    if response.status_code == 200:
        return float(response.text)
    else:
        raise Exception("Failed to retrieve temperature")

# Function to get humidity from ESP32-CAM
def get_humidity():
    response = requests.get(humidity_url)
    if response.status_code == 200:
        return float(response.text)
    else:
        raise Exception("Failed to retrieve humidity")
    
# Call the function to get the first frame
get_frame()
while True:
    # Get the next frame from the camera
    get_frame()
    # If we didn't get a frame, skip to the next iteration of the loop
    if frame is None:
        continue

    #Calculating fps
    counter += 1
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # perform the inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * 720)))
            xmin = int(max(1,(boxes[i][1] * 1280)))
            ymax = int(min(720,(boxes[i][2] * 720)))
            xmax = int(min(1280,(boxes[i][3] * 1280)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1) # Draw label text       
            
            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(fps),(30,50),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
            # Example usage of retrieving temperature and humidity
            try:
                temperature = get_temperature()
                humidity = get_humidity()
                # Draw temperature and humidity on the frame
                text = f"Humidity: {humidity}%  Temperature: {temperature}degC"
                cv2.putText(frame, text, (frame.shape[1] - 365, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"Error: {str(e)}")  
                
            # All the results have been drawn on the frame, so it's time to display it.
            
            if object_name == 'person':
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f'{object_name}_{timestamp}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, frame)
                
            if object_name == 'cell phone':
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename1 = f'{object_name}_{timestamp}.jpg'
                output_path1 = os.path.join(output_folder1, output_filename1)
                cv2.imwrite(output_path1, frame)
                
            if object_name == 'cat':
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename2 = f'{object_name}_{timestamp}.jpg'
                output_path2 = os.path.join(output_folder2, output_filename2)
                cv2.imwrite(output_path2, frame)
    
    
    cv2.imshow('Live Stream', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
