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
RAW_SEGMAP_ROOT = 'data/OBIMD_align/facsimile'
REFINE_SEGMAP_ROOT = 'data/OBIMD_align/facsimile_refine'
EXPAND = 0.1
GLOBAL_BOX_PROMPT = True

with open(LABEL_JSON) as f:
    datas = json.load(f)
with open('data/SA-V_sample/train.txt') as f:
    train_list = [line.strip().split('.')[0] for line in f.readlines()]
os.makedirs(OUTPUT_DIR, exist_ok=True)
char_id = 0
for image_id, data in tqdm(enumerate(datas)):
    image_name = os.path.basename(data['Rubbing'])
    facsimile_raw = cv2.imread(os.path.join(RAW_SEGMAP_ROOT, image_name), flags=cv2.IMREAD_GRAYSCALE)
    facsimile_refine = cv2.imread(os.path.join(REFINE_SEGMAP_ROOT, image_name), flags=cv2.IMREAD_GRAYSCALE)
    if facsimile_raw is None or facsimile_refine is None:
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
                if np.any(facsimile_raw[y:min(y+h, H), x:min(x+w, W)] > 0) and np.any(facsimile_refine[y:min(y+h, H), x:min(x+w, W)] > 0):
                    oracle_chars.append(char) # get available chars
            except:
                pass
    
    video_annos = dict()
    video_annos['video_id'] = image_id
    video_annos['video_duration'] = 2/24
    video_annos['video_frame_count'] = 2
    video_annos['video_height'] = H
    video_annos['video_width'] = W
    video_annos['video_resolution'] = W * H
    video_annos['video_environment'] = 'Indoor'
    video_annos['video_split'] = 'train' if image_name.split('.')[0] in train_list else 'test'
    video_annos['masklet'] = [[],[]]
    video_annos['masklet_id'] = []
    video_annos['masklet_size_rel'] = []
    



    for char in oracle_chars:
        mask = np.zeros_like(facsimile_raw)
        x, y, w, h = list(map(int, char['Position'].split(',')))
        x = max(0, int(x - EXPAND * w))
        y = max(0, int(y - EXPAND * h))
        w = int(w + 2 * EXPAND * w)
        h = int(h + 2 * EXPAND * h)
        mask[y:min(y+h, H), x:min(x+w, W)] = facsimile_raw[y:min(y+h, H), x:min(x+w, W)] # TOFIX: check area?
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