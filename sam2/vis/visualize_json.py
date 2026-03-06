# 可视化
import os
import cv2
import json
from pycocotools import mask as maskUtils
import numpy as np
import random
import matplotlib.pyplot as plt

json_dir = "data/OBIMD_test100/facsimile_json"
rubbing_dir = "data/OBIMD_test100/rubbing"
facsimile_dir = "data/OBIMD_test100/facsimile"
vis_dir = "data/OBIMD_test100/visualization"

random.seed(42)
file_list = random.sample(os.listdir(json_dir), 100) # fixed for same generation
# def get_colors(num_colors):
#     colors = []
#     for i in range(num_colors):
#         hue = int(180 * i / num_colors)
#         saturation = 255
#         value = 255
#         color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
#         colors.append(tuple(map(int, color)))
#     return colors

for file in file_list:
    image_path = os.path.join(rubbing_dir, file.replace('json', 'jpg'))
    facsimile = cv2.imread(os.path.join(facsimile_dir, file.replace('json', 'jpg')))
    image = cv2.imread(image_path)
    
    with open(os.path.join(json_dir, file)) as f:
        data = json.load(f)
    mask = np.zeros_like(image, dtype=np.uint8)
    num_annotations = len(data['annotations'])
    # colors = get_colors(num_annotations)
    
    for i, annotation in enumerate(data['annotations']):
        bbox = annotation['bbox']
        x0, y0, w, h = bbox
        color = np.random.randint(0, 256, size=3).tolist()
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        
        # try:
        #     iou_text = f"IoU: {annotation['predicted_iou']:.2f}"
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.5
        #     font_thickness = 1
        #     text_size = cv2.getTextSize(iou_text, font, font_scale, font_thickness)[0]
        #     text_x = int(x0 + (w - text_size[0]) / 2)
        #     text_y = int(y0 - 5)  # Position above the box
        #     cv2.putText(image, iou_text, (text_x, text_y), font, font_scale, color, font_thickness)
        # except:
        #     pass
        
        m = maskUtils.decode(annotation['segmentation'])
        color_mask = np.zeros_like(mask)
        color_mask[m == 1] = color
        mask = cv2.bitwise_or(mask, color_mask)

    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, file.replace('json', 'jpg'))
    
    overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    combined = np.hstack((image, overlay, facsimile))
    cv2.imwrite(output_path, overlay)