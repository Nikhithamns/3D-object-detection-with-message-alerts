import cv2
import numpy as np
import os
import time
import logging
import pywhatkit as kit

# Set up logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO, format='%(asctime)s:%(message)s')

# Configuration
FACE_CASCADE_PATH = '/Users/nikhithamns/Downloads/Object recognition/GT8/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
YOLO_CONFIG_FILE = '/Users/nikhithamns/Downloads/yolov3.cfg'
YOLO_WEIGHTS_FILE = '/Users/nikhithamns/Downloads/yolov3.weights'
NAMES_FILE = '/Users/nikhithamns/Downloads/coco.names'
WHATSAPP_PHONE_NUMBER = '+917013342133'  

# Verify that the YOLO files exist
if not os.path.exists(YOLO_CONFIG_FILE):
    raise FileNotFoundError(f"Config file not found: {YOLO_CONFIG_FILE}")
if not os.path.exists(YOLO_WEIGHTS_FILE):
    raise FileNotFoundError(f"Weights file not found: {YOLO_WEIGHTS_FILE}")
if not os.path.exists(NAMES_FILE):
    raise FileNotFoundError(f"Names file not found: {NAMES_FILE}")

# Load class names
with open(NAMES_FILE, 'r') as f:
    classes = f.read().strip().split('\n')

# Print loaded classes for debugging
print(f"Loaded classes: {classes}")

# Load YOLO model
try:
    net = cv2.dnn.readNet(YOLO_WEIGHTS_FILE, YOLO_CONFIG_FILE)
except cv2.error as e:
    print(f"Error loading YOLO files: {e}")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Initialize video capture
video = cv2.VideoCapture(0)  # Change to the path of your video file for video file input
labels = []
start_time = time.time()

def send_whatsapp_message(message):
    try:
        now = time.localtime()
        hour = now.tm_hour
        minute = now.tm_min + 2
        if minute >= 60:
            minute -= 60
            hour += 1
        kit.sendwhatmsg(WHATSAPP_PHONE_NUMBER, message, hour, minute)
        print("WhatsApp message sent successfully.")
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {e}")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform object detection with YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []
    h, w = frame.shape[:2]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    bbox = []
    label_ids = []
    conf = []

    if len(indices) > 0:
        for i in indices.flatten():
            bbox.append(boxes[i])
            label_ids.append(class_ids[i])
            conf.append(confidences[i])

    # Convert class IDs to class names
    label = [classes[id] if id < len(classes) else "Unknown" for id in label_ids]

    # Log detected labels
    for item, confidence in zip(label, conf):
        if item not in labels and item != "Unknown":
            labels.append(item)
            logging.info(f"Detected: {item} with confidence: {confidence}")

    # Print detected labels for debugging
    print(f"Detected labels: {label}")

    # Draw bounding boxes
    for box, lbl, confidence in zip(bbox, label, conf):
        x, y, dw, dh = box
        cv2.rectangle(frame, (x, y), (x+dw, y+dh), (0, 255, 0), 2)
        cv2.putText(frame, f'{lbl} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw face bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    start_time = time.time()
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    # Terminate object detection on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# Construct the sentence from detected labels
if labels:
    detected_objects_message = ", and ".join(f"a {lbl}" for lbl in labels) + "."
    print(detected_objects_message)
    send_whatsapp_message(f"I found {detected_objects_message}")
else:
    print("No objects detected.")
