import torch
from ultralytics import YOLO
from ultralytics import settings
import cv2
#画框
def drawBox(data,image,name):
    x1, y1, x2, y2, conf, id = data
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
    cv2.putText(image, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    return image
detection_classes=[] 
#设置路径
settings.update({'runs_dir':'./'})
#加载模型
model = YOLO('yolov8n.yaml')#加载新模型
model.load('yolov8n.pt')
#训练模型
result=model.train(data='coco128.yaml',epochs=10,imgsz=640,device=0)
# 验证模型
metrics = model.val()  # 无需参数，数据集和设置记忆
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 包含每个类别的map50-95列表

#加载opencv视频流

video  = cv2.VideoCapture()
while True:
        (grabbed,frame) = video.read()
        if not grabbed:
            break
        results = model.predict(frame,stream=False)
        cv2.imshow('image', frame)
        cv2.waitKey(500)

results = model.predict(frame,stream=False)
for result in results:
     for data in result.boxes.data.tolist():
           #print(data)
           id = data[5]
           drawBox(data, frame,detection_classes[id])