# 可视化
import os
import cv2
import json
from pycocotools import mask as maskUtils
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil
from sam2.data_utils.utils import load_null_annotations

folder = ['OBIMD_raw_hj', 'OBIMD_segformer', 'OBIMD_charformer', 'OBIMD_stage3_selfiter', 'OBIMD_iou0.6/stage4']
json_dir = "data/{}/facsimile_json"
rubbing_dir = "data/OBIMD_raw_hj/rubbing"
facsimile_dir = "data/OBIMD_raw_hj/facsimile"
vis_dir = "visualization"


file = random.sample(os.listdir(json_dir.format(folder[2])), 1)[0] # fixed for same generation
def get_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        saturation = 255
        value = 255
        color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

shutil.copy(os.path.join(rubbing_dir, file.replace('json', 'jpg')), os.path.join(vis_dir, 'raw_' + file.replace('json', 'jpg')))
colors = get_colors(20)
null_lookup = load_null_annotations('data/OBIMD_raw_hj/label.json')
for fold in folder:
    random.seed(42)
    image_path = os.path.join(rubbing_dir, file.replace('json', 'jpg'))
    facsimile = cv2.imread(os.path.join(facsimile_dir, file.replace('json', 'jpg')))
    image = cv2.imread(image_path)
    
    with open(os.path.join(json_dir.format(fold), file)) as f:
        data = json.load(f)
    mask = np.zeros_like(image, dtype=np.uint8)
    if fold == 'OBIMD_raw_hj':
        anns = []
        for ann in data["annotations"]:
            if ','.join([str(x) for x in ann['crop_box']]) not in [c['Position'] for c in null_lookup[file.split('.')[0]]]:
                anns.append(ann)
        data['annotations'] = anns
    num_annotations = len(data['annotations'])
    annotation = sorted(data['annotations'], key=lambda x: x['bbox'][1], reverse=True)
    
    for i, annotation in enumerate(annotation):
        bbox = annotation['bbox']
        x0, y0, w, h = bbox
        color = colors[i]
        m = maskUtils.decode(annotation['segmentation'])
        color_mask = np.zeros_like(mask)
        color_mask[m == 1] = color
        mask = cv2.bitwise_or(mask, color_mask)

    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, fold.replace('/', '_') + file.replace('json', 'jpg'))
    print(output_path)
    
    overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    combined = np.hstack((image, overlay, facsimile))
    cv2.imwrite(output_path, overlay)