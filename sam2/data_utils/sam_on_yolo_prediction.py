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
from sam2.data_utils.utils import calculate_stability_score

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_align.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
mask_threshold = 0
stability_score_offset = 1.0 
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor = SAM2ImagePredictor(sam2_model)
    char_id = 0
    for image_id, img_path in tqdm(enumerate(sorted(os.listdir('data/OBIMD_HJ_diff/rubbings')))):
        image = cv2.imread(os.path.join('data/OBIMD_HJ_diff/rubbings', img_path))
        H, W, C = image.shape
        if not os.path.exists(os.path.join('data/OBIMD_HJ_diff/yolo_prediction', img_path.replace('.jpg', '.txt'))):
            continue
        if os.path.exists(os.path.join('data/OBIMD_HJ_diff/fascimile_json_align', f'{img_path.split(".")[0]}.json')):
            continue
        input_box = []
        with open(os.path.join('data/OBIMD_HJ_diff/yolo_prediction', img_path.replace('.jpg', '.txt'))) as f:
            for line in f:
                box = list(map(float, line.strip().split()[1:]))
                x,y,w,h = box
                x0, y0 = (x - w/2) * W, (y - h/2) * H
                x1, y1 = (x + w/2) * W, (y + h/2) * H
                input_box.append([x0, y0, x1, y1])
        predictor.set_image(image)
        input_box = np.array(input_box).reshape(-1, 4)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
            return_logits=True
        )

        stability_scores = calculate_stability_score(masks)
        masks = (masks > 0).astype(np.uint8) * 255
        if masks.shape[0] == 1:
            masks = masks[None, ...]  # Ensure masks is a batch of masks

        pred_result = dict(image_info=dict(image_id=image_id, width=W, height=H, file_name=img_path), annotations=[])
        for mask, score, box, stab_score in zip(masks, scores, input_box, stability_scores):
            mask = np.asfortranarray(mask[0])
            rle = maskUtils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle)
            box[2:] = box[2:] - box[:2]  # Convert from [x0, y0, x1, y1] to [x, y, w, h]
            crop_box = box.tolist()
            pred_result['annotations'].append(dict(id=char_id, bbox=bbox.astype(int).tolist(), area=int(area), 
                                                   segmentation=rle,
                                                   predicted_iou=float(score), 
                                                   stability_score=float(stab_score),
                                                   crop_box=crop_box))
            char_id += 1
        with open(os.path.join('data/OBIMD_HJ_diff/fascimile_json_align', f'{img_path.split(".")[0]}.json'), 'w') as f:
            json.dump(pred_result, f, indent=2, ensure_ascii=False)