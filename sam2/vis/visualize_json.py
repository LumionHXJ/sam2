# 可视化
import os
import cv2
import json
from pycocotools import mask as maskUtils
import numpy as np
import random
import matplotlib.pyplot as plt

json_dir = "data/OBIMD_stage2/facsimile_json"
rubbing_dir = "data/OBIMD_stage2/rubbing"
facsimile_dir = "data/OBIMD_stage2/facsimile_no_border"
vis_dir = "data/OBIMD_stage2/visualization"

random.seed(42)
file_list = random.sample(os.listdir(json_dir), 100) # fixed for same generation
for file in file_list:
    image_path = os.path.join(rubbing_dir, file.replace('json', 'jpg'))
    facsimile = cv2.imread(os.path.join(facsimile_dir, file.replace('json', 'jpg')))
    image = cv2.imread(image_path)
    with open(os.path.join(json_dir, file)) as f:
        data = json.load(f)
    mask = np.zeros_like(image[..., 0], dtype=np.uint8)
    for annotation in data['annotations']:
        if annotation['area'] == 0:
            print(file, annotation)
        bbox = annotation['bbox']
        x0, y0, w, h = bbox
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        try:
            iou_text = f"IoU: {annotation['predicted_iou']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(iou_text, font, font_scale, font_thickness)[0]
            text_x = int(x0 + (w - text_size[0]) / 2)
            text_y = int(y0 - 5)  # Position above the box
            cv2.putText(image, iou_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
        except:
            pass
        m = maskUtils.decode(annotation['segmentation'])
        mask = cv2.bitwise_or(mask, m.astype(np.uint8) * 255)

    # Convert facsimile to OpenCV format and save to temp directory
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, file.replace('json', 'jpg'))
    # Apply a colormap to the facsimile for better visualization
    colored_facsimile = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, colored_facsimile, 0.5, 0)
    combined = np.hstack((overlay, facsimile))
    cv2.imwrite(output_path, combined)