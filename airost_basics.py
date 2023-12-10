from ultralytics import YOLO
import numpy
# By detecting through an image and outputs.
# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# predict on an image
detection_output = model.predict(source="images/train/001.jpeg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
