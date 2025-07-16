import numpy as np
import json
from tqdm import tqdm
import os
import cv2

def calc_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

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
                if x < 0 or y < 0 or x + w > W or y + h > H:
                    continue
                oracle_chars.append(char)
        data_lookup[image_name.split('.')[0]] = oracle_chars
    return data_lookup

def load_null_annotations(data_path):
    with open(data_path) as f:
        datas = json.load(f)
    data_lookup = dict()
    for image_id, data in enumerate(datas):
        image_name = os.path.basename(data['Rubbing'])
        # CHECK SHAPE etc.
        oracle_chars = []
        for sentence in data['RecordUtilSentenceGroupVoList']:
            for char in sentence["RecordUtilOracleCharVoList"]:
                if char['Label'] is None:
                    oracle_chars.append(char) 
        data_lookup[image_name.split('.')[0]] = oracle_chars
    return data_lookup

def filter_box_by_iou(box_to_filter, ignore_boxes, thr=0.5):
    """如果和ignore_boxes的box有iou大于thr的，则返回True"""
    for ignore in ignore_boxes:
        if calc_box_iou(box_to_filter, ignore) > thr:
            return True
    return False