# 进行目标检测

from ultralytics import YOLO
import time
from tqdm import tqdm
import os

model = YOLO("/home/zhanglizhong/yolov12-main/workdir_f/yolov12/weights/best.pt", verbose=False)
for img_path in tqdm(os.listdir('data/OBIMD_HJ_diff/rubbings/')):
    if os.path.exists(os.path.join('data/OBIMD_HJ_diff/yolo_prediction', img_path.replace('jpg', 'txt'))):
        continue
    result = model(os.path.join('data/OBIMD_HJ_diff/rubbings/', img_path))[0]
    result.save_txt(os.path.join('data/OBIMD_HJ_diff/yolo_prediction', img_path.replace('jpg', 'txt')))