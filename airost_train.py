from ultralytics import YOLO

#Loading a model
model = YOLO("weights/yolov8n.pt")
# model = YOLO("yolov8n.yaml") # Build a new model (Optional)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')

#Training the models mentioned above
# config.yaml is where the images you want the model to be trained. 
results = model.train(data = "config.yaml", epochs = 10)
