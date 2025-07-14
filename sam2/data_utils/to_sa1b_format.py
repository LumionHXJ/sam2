import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm

LABEL_JSON = 'data/OBIMD_align/label.json'
IMAGE_ROOT = 'data/OBIMD_align/rubbing'
OUTPUT_DIR = 'data/OBIMD_align/facsimile_json'
SEGMAP_ROOT = 'data/OBIMD_align/facsimile'
EXPAND = 0.1
GLOBAL_BOX_PROMPT = False

with open(LABEL_JSON) as f:
    datas = json.load(f)
os.makedirs(OUTPUT_DIR, exist_ok=True)
char_id = 0
for image_id, data in tqdm(enumerate(datas)):
    image_name = os.path.basename(data['Rubbing'])
    facsimile = cv2.imread(os.path.join(SEGMAP_ROOT, image_name.replace('jpg', 'png')), flags=cv2.IMREAD_GRAYSCALE)
    if facsimile is None:
        continue
    # CHECK SHAPE etc.
    W, H = Image.open(os.path.join(IMAGE_ROOT, image_name)).size
    image_info = dict(image_id=image_id, width=W, height=H, file_name=image_name)
    annotations = []
    oracle_chars = []
    for sentence in data['RecordUtilSentenceGroupVoList']:
        for char in sentence["RecordUtilOracleCharVoList"]:
            x, y, w, h = list(map(int, char['Position'].split(',')))
            try:
                if np.any(facsimile[y:min(y+h, H), x:min(x+w, W)] > 0):
                    oracle_chars.append(char) # get available chars
            except:
                pass
    for char in oracle_chars:
        mask = np.zeros_like(facsimile)
        x, y, w, h = list(map(int, char['Position'].split(',')))
        x = max(0, int(x - EXPAND * w))
        y = max(0, int(y - EXPAND * h))
        w = int(w + 2 * EXPAND * w)
        h = int(h + 2 * EXPAND * h)
        mask[y:min(y+h, H), x:min(x+w, W)] = facsimile[y:min(y+h, H), x:min(x+w, W)] # TOFIX: check area?
        for _char in oracle_chars:
            if char == _char:
                continue
            _x, _y, _w, _h = list(map(int, _char['Position'].split(',')))
            mask[_y:min(_y+_h, H), _x:min(_x+_w, W)] = 0 # remove other chars
        mask = np.asfortranarray(mask)
        rle = maskUtils.encode(mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        area = maskUtils.area(rle)
        bbox = maskUtils.toBbox(rle)
        annotations.append(dict(id=char_id, bbox=bbox.astype(int).tolist(), area=int(area), segmentation=rle))
        char_id += 1
    if GLOBAL_BOX_PROMPT:
        # 加入一个全局的背景prompt，为了不和中心区域冲突需要取反
        mask = np.asfortranarray(255 - facsimile)
        rle = maskUtils.encode(mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        area = maskUtils.area(rle)
        bbox = [0, 0, W, H]
        annotations.append(dict(id=char_id, bbox=bbox, area=int(area), segmentation=rle))
        char_id += 1
    with open(os.path.join(OUTPUT_DIR, f'{image_name.split(".")[0]}.json'), 'w') as f:
        json.dump(dict(image_info=image_info, annotations=annotations), f, indent=2)