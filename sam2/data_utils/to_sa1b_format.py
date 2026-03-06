import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm
from sam2.data_utils.utils import load_raw_annotations, extract_connected_component_opencv

LABEL_JSON = 'data/OBIMD_raw_hj/label.json'
IMAGE_ROOT = 'data/OBIMD_raw_hj/rubbing'
OUTPUT_DIR = 'data/OBIMD_raw_hj/facsimile_json'
SEGMAP_ROOT = 'data/OBIMD_raw_hj/facsimile'
EXPAND = 0

with open(LABEL_JSON) as f:
    datas = json.load(f)
os.makedirs(OUTPUT_DIR, exist_ok=True)
char_id = 0
data_lookup = load_raw_annotations('data/OBIMD_raw_hj/label.json', SEGMAP_ROOT, ignore_null=True)
for image_id, data in tqdm(enumerate(datas)):
    image_name = os.path.basename(data['Rubbing'])
    # CHECK SHAPE etc.
    if not os.path.exists(os.path.join(SEGMAP_ROOT, image_name)):
        continue
    facsimile = cv2.imread(os.path.join(SEGMAP_ROOT, image_name), flags=cv2.IMREAD_GRAYSCALE)
    W, H = Image.open(os.path.join(IMAGE_ROOT, image_name)).size
    image_info = dict(image_id=image_id, width=W, height=H, file_name=image_name)
    annotations = []
    oracle_chars = []
    for sentence in data['RecordUtilSentenceGroupVoList']:
        for char in sentence["RecordUtilOracleCharVoList"]:
            x, y, w, h = list(map(int, char['Position'].split(',')))
            try:
                if np.any(facsimile[y:min(y+h, H), x:min(x+w, W)] > 0):
                    oracle_chars.append(char) # 忽略空白字符
            except:
                continue
    if len(oracle_chars) == 0:
        continue
    for char in oracle_chars:
        mask = np.zeros_like(facsimile)
        x, y, w, h = list(map(int, char['Position'].split(',')))
        x = max(0, int(x - EXPAND * w))
        y = max(0, int(y - EXPAND * h))
        w = int(w + 2 * EXPAND * w)
        h = int(h + 2 * EXPAND * h)
        mask[y:min(y+h, H), x:min(x+w, W)] = facsimile[y:min(y+h, H), x:min(x+w, W)]
        for _char in data_lookup[image_name.split('.')[0]]:
            if char['Position'] == _char['Position']:
                continue
            _x, _y, _w, _h = list(map(int, _char['Position'].split(',')))
            roi_y1, roi_y2 = _y, min(_y+_h, H)
            roi_x1, roi_x2 = _x, min(_x+_w, W)
            roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
            if np.any(roi > 0):
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi.astype(np.uint8), connectivity=8)
                for label in range(1, num_labels):
                    ys, xs = np.where(labels == label)
                    orig_x = xs[0] + roi_x1
                    orig_y = ys[0] + roi_y1
                    _, component_mask = extract_connected_component_opencv(mask, start_x=orig_x, start_y=orig_y)
                    component_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 0
                    if not np.any(component_mask):
                        mask[roi_y1:roi_y2, roi_x1:roi_x2][labels == label] = 0
        mask = np.asfortranarray(mask)
        rle = maskUtils.encode(mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        area = maskUtils.area(rle)
        if area == 0:
            continue
        bbox = maskUtils.toBbox(rle)
        annotations.append(dict(id=char_id, bbox=bbox.astype(int).tolist(), area=int(area), segmentation=rle, crop_box=[x, y, w, h]))
        char_id += 1
    if len(annotations) == 0:
        continue
    with open(os.path.join(OUTPUT_DIR, f'{image_name.split(".")[0]}.json'), 'w') as f:
        json.dump(dict(image_info=image_info, annotations=annotations), f, indent=2)