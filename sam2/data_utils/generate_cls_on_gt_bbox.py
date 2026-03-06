import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.data_utils.utils import get_tight_box

sam2_checkpoint = "sam2_logs/configs/sam2.1_baseline/sam2.1_hiera_l_OBIMD_charformer.yaml/checkpoints/checkpoint_10.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l_highres.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:1')
predictor = SAM2ImagePredictor(sam2_model)
exp_name = 'charformer'
output_dir = f'data/OBIMD_cls/{exp_name}'
os.makedirs(output_dir, exist_ok=False)

with open('data/OBIMD_cls/class.txt', 'r') as f:
    topclass = set([l.strip() for l in f.readlines()])
with open("data/OBIMD_raw_hj/test.txt") as f:
    test_list = set([l.strip() for l in f.readlines()])
with open("data/OBIMD_raw_hj/label_filt_train.json") as f:
    datas = json.load(f)

for label in topclass:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True) # 每个类别一个文件夹，否则推理的时候对不上

for image_id, data in tqdm(enumerate(datas)):
    image_path = os.path.basename(data['Rubbing'])
    if image_path.split('.')[0] not in test_list:
        continue
    
    input_box = []
    labels = []
    for sentence in data['RecordUtilSentenceGroupVoList']:
        for char in sentence["RecordUtilOracleCharVoList"]:
            if char['Label'] in topclass:
                x, y, w, h = list(map(int, char['Position'].split(',')))
                input_box.append([x, y, x + w, y + h])
                labels.append(char['Label'])
    input_box = np.array(input_box).reshape(-1, 4)
    if len(input_box) == 0:
        continue

    image = cv2.imread(os.path.join('data/OBIMD_raw_hj/rubbing', image_path))
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box, # use gt box here
        multimask_output=False,
    )
    if len(masks) == 1:
        masks = masks[None]
    masks = masks.astype(np.uint8) * 255

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        if not mask.any():
            continue
        os.makedirs(f'{output_dir}/{label}', exist_ok=True)
        x, y, w, h = get_tight_box(mask[0])
        cv2.imwrite(f'{output_dir}/{label}/{image_path.split(".")[0]}_{idx}.png', mask[0][y:y+h, x:x+w])