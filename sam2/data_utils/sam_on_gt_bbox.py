# 自监督飞轮阶段，使用yolov12的检测结果作为box prompt，进行打标

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import json
from tqdm import tqdm
import os
from pycocotools import mask as maskUtils
import torch
from sam2.data_utils.utils import load_raw_annotations

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage1.yaml/checkpoints/checkpoint_10.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
mask_threshold = 0
stability_score_offset = 1.0 
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor = SAM2ImagePredictor(sam2_model)
    char_id = 0 # TOFIX: 暂时没有断点
    # load train_list
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = [line.strip() for line in f.readlines()]

    # load raw data
    data_lookup = load_raw_annotations("data/OBIMD_raw_hj/label_filtered.json", "data/OBIMD_raw_hj/facsimile", ignore_null=True)

    for image_id, path in tqdm(enumerate(train_list)):
        image = cv2.imread(os.path.join('data/OBIMD_stage2/rubbing', path+'.jpg'))
        H, W, C = image.shape
        if os.path.exists(os.path.join('data/OBIMD_stage2/rubbing/facsimile_json', path+'.json')):
            continue
        input_box = []
        for char in data_lookup[path]:
            x, y, w, h = list(map(int, char['Position'].split(',')))
            input_box.append([x, y, x + w, y + h])
        predictor.set_image(image)
        input_box = np.array(input_box).reshape(-1, 4)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        masks = (masks * 255).astype(np.uint8)
        if masks.shape[0] == 1:
            masks = masks[None, ...]  # Ensure masks is a batch of masks

        pred_result = dict(image_info=dict(image_id=image_id, width=W, height=H, file_name=path+'.jpg'), annotations=[])
        for mask, score, box in zip(masks, scores, input_box):
            mask = np.asfortranarray(mask[0])
            rle = maskUtils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle)
            box[2:] = box[2:] - box[:2]  # Convert from [x0, y0, x1, y1] to [x, y, w, h]
            crop_box = box.tolist()
            pred_result['annotations'].append(dict(id=char_id, bbox=bbox.astype(int).tolist(), area=int(area), segmentation=rle, 
                                    predicted_iou=float(score), crop_box=crop_box))
            char_id += 1
        with open(os.path.join('data/OBIMD_stage2/facsimile_json', path+'.json'), 'w') as f:
            json.dump(pred_result, f, indent=2, ensure_ascii=False)