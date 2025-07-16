import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from pycocotools import mask as maskUtils
from tqdm import tqdm

SA1B_ROOT = 'data/OBIMD_stage2/facsimile_json_iou0.65'
IMAGE_ROOT = 'data/OBIMD_stage2/rubbing'
OUTPUT_DIR = 'data/OBIMD_stage2/facsimile_sav_iou0.65'
RAW_SEGMAP_ROOT = 'data/OBIMD_raw_hj/facsimile'

with open('data/OBIMD_stage2/train.txt') as f:
    train_list = [line.strip().split('.')[0] for line in f.readlines()]
os.makedirs(OUTPUT_DIR, exist_ok=True)
char_id = 0
for image_id, file in tqdm(enumerate(train_list)):
    with open(os.path.join(SA1B_ROOT, f'{file}.json')) as f:
        sa1b_data = json.load(f)
    facsimile= cv2.imread(os.path.join(RAW_SEGMAP_ROOT, f'{file}.jpg'), flags=cv2.IMREAD_GRAYSCALE)
    mask = np.asfortranarray(facsimile)
    rle = maskUtils.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    H, W = facsimile.shape[:2]
    
    video_annos = dict()
    video_annos['video_id'] = image_id
    video_annos['video_duration'] = 2/24
    video_annos['video_frame_count'] = 2
    video_annos['video_height'] = H
    video_annos['video_width'] = W
    video_annos['video_resolution'] = W * H
    video_annos['video_environment'] = 'Indoor'
    video_annos['video_split'] = 'train'
    video_annos['masklet'] = [[],[]]
    video_annos['masklet_id'] = []
    video_annos['masklet_size_rel'] = []
    video_annos['masklet_size_abs'] = []
    video_annos['masklet_size_bucket'] = []
    video_annos['masklet_visibility_changes'] = []
    video_annos['masklet_first_appeared_frame'] = []
    video_annos['masklet_frame_count'] = []
    video_annos['masklet_type'] = []
    video_annos['masklet_stability_score'] = [[], []]
    video_annos['masklet_num'] = len(sa1b_data['annotations'])

    for ann in sa1b_data['annotations']:
        video_annos['masklet'][0].append(rle) # 第一帧是
        video_annos['masklet'][1].append(ann['segmentation'])
        video_annos['masklet_id'].append(char_id)
        char_id += 1
        video_annos['masklet_size_rel'].append(ann['area'] / (W * H))
        video_annos['masklet_size_abs'].append(ann['area'])
        video_annos['masklet_size_bucket'].append("medium")
        video_annos['masklet_visibility_changes'].append(0)
        video_annos['masklet_first_appeared_frame'].append(0)
        video_annos['masklet_frame_count'].append(2)
        video_annos['masklet_type'].append("auto")
        video_annos['masklet_stability_score'][0].append(1)
        video_annos['masklet_stability_score'][1].append(ann['predicted_iou'])
    with open(os.path.join(OUTPUT_DIR, f'{file}_auto.json'), 'w') as f:
        json.dump(video_annos, f, ensure_ascii=False, indent=4)