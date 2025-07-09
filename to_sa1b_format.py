import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm

with open('data/label.json') as f:
    datas = json.load(f)
image_root = 'data/OBIMD_rubbing'
output_dir = 'data/OBIMD_facsimile_json'
char_id = 0
for image_id, data in tqdm(enumerate(datas)):
    image_name = os.path.basename(data['Rubbing'])
    facsimile = cv2.imread(os.path.join('data/OBIMD_facsimile_no_boarder', image_name), flags=cv2.IMREAD_GRAYSCALE)
    # CHECK SHAPE etc.
    W, H = Image.open(os.path.join(image_root, image_name)).size
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
    with open(os.path.join(output_dir, f'{image_name.split(".")[0]}.json'), 'w') as f:
        json.dump(dict(image_info=image_info, annotations=annotations), f, indent=2)