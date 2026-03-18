from ultralytics import YOLO
from datetime import datetime

model = YOLO("/home/huxingjian/model/MarsOCR/yolo/models/yolov12x.pt")
# Train the model
train_results = model.train(
    data="sam2/configs/yolo/OBIMD.yaml",  # path to dataset YAML
    task='detect',
    epochs=50,  # number of training epochs
    imgsz=1024,  # training image size
    batch=12, # split on all gpus
    device="0,2,3",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    name=datetime.now().strftime("%y%m%d-%H%M%S"),
    cfg="sam2/configs/yolo/overwrite.yaml",
)

