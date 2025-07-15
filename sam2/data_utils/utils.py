import numpy as np
import json
from tqdm import tqdm
import os
import cv2

def load_raw_annotations(data_path, segmap_root, ignore_null=False):
    with open(data_path) as f:
        datas = json.load(f)
    data_lookup = dict()
    for image_id, data in tqdm(enumerate(datas)):
        image_name = os.path.basename(data['Rubbing'])
        facsimile = cv2.imread(os.path.join(segmap_root, image_name), flags=cv2.IMREAD_GRAYSCALE)
        if facsimile is None:
            print(image_name)
            continue
        # CHECK SHAPE etc.
        H, W = facsimile.shape[:2]
        oracle_chars = []
        for sentence in data['RecordUtilSentenceGroupVoList']:
            for char in sentence["RecordUtilOracleCharVoList"]:
                if ignore_null and char['Label'] is None:
                    continue
                x, y, w, h = list(map(int, char['Position'].split(',')))
                try:
                    if np.any(facsimile[y:min(y+h, H), x:min(x+w, W)] > 0):
                        oracle_chars.append(char) # get available chars
                except:
                    pass
        data_lookup[image_name.split('.')[0]] = oracle_chars
    return data_lookup