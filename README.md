# Object detection using YOLOv8 and opencv

Object detection by using YOLOv8 and opencv-python. 
The object detector is using a pretrained-model provided by the [Ultralytics Page](https://github.com/ultralytics/ultralytics). 

To get started with all of these, you need to download git bash from the website [Git](https://git-scm.com/downloads).

## Documentation 
After you downloaded the git, you can copy the commands below and `git bash` on the directory file you want.
```bash
git clone https://github.com/notEhEnG/object_detection_yolov8.git
```
Next, you should have `python version 3.10` installed.
You can download the python [here](https://www.python.org/downloads/release/python-3100/). 

Suggestion:
It's better to use an IDE (Integrated Development Environment) such as `PyCharm` for you to be using these libraries provided below in the requirements.txt. 
You can download the PyCharm [here](https://www.jetbrains.com/edu-products/download/other-PCE.html).
```py
pip install -r requirements.txt 
```

## Coding
```py
from ultralytics import YOLO

#Loading a model
model = YOLO("weights/yolov8n.pt")
# model = YOLO("yolov8n.yaml") # Build a new model (Optional)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')

#Training the models mentioned above
# config.yaml is where the images you want the model to be trained. 
results = model.train(data = "config.yaml", epochs = 10)
```
The codes provided above is to train a `new model` / `pretrained-model`. 

```py
import cv2
import numpy as np
import random
from ultralytics import YOLO

# opening the file in read mode
my_file = open("utils/classes_name.txt", "r")

# reading the file
data = my_file.read()

# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")

my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
# In this case, you can also used a new trained model from scratch. **PATH to the best.pt**

model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 640

# cap = cv2.VideoCapture(0) for external camera
# cap = cv2.VideoCapture(VIDEOPATH), where VIDEOPATH must be same in the root dir
cap = cv2.VideoCapture("inference/videos/bird.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(frame,
                          (int(bb[0]), int(bb[1])),
                          (int(bb[2]), int(bb[3])),
                          detection_colors[int(clsID)],
                          3)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,
                        class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```
