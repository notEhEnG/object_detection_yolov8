from ultralytics import YOLO

#Loading a model
model = YOLO("weights/yolov8n.pt")
model = YOLO("yolov8n.yaml") #Build a new model from scratch
# model = YOLO("weights/yolov8n.pt")
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

#Training the models mentioned above
results = model.train(data = "config.yaml", epochs = 10)