from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import json
from tqdm import tqdm
import os
from pycocotools import mask as maskUtils
import torch
import shutil

# checkpoint from last round
sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam_flywheel_round1/checkpoints/checkpoint_1.pt"
last_round_data = 'data/OBIMD_HJ_diff/fascimile_json_init'
output_dir = 'data/OBIMD_HJ_diff/fascimile_json_round1_out'
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
mask_threshold = 0
stability_score_offset = 1.0 
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:2')
iou_threshold_last_round = 0.7 # last data filter (only try to correct this part)

os.makedirs(output_dir, exist_ok=True)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor = SAM2ImagePredictor(sam2_model)
    for file in tqdm(sorted(os.listdir(last_round_data))):
        if os.path.exists(os.path.join(output_dir, file)):
            continue
        with open(os.path.join(last_round_data, file), 'r') as f:
            pred_result = json.load(f)
        input_box = []
        for ann in pred_result['annotations']:
            # if annotations is well predicted
            if ann['predicted_iou'] >= iou_threshold_last_round:
                continue
            input_box.append(ann['crop_box'])
        if len(input_box) == 0:
            shutil.copy(os.path.join(last_round_data, file), os.path.join(output_dir, file))
            continue # no result to correct
        input_box = np.array(input_box).reshape(-1, 4)
        input_box[:, 2:] += input_box[:, :2]  # Convert from [x, y, w, h] to [x0, y0, x1, y1]

        img_path = file.replace('.json', '.jpg')
        image = cv2.imread(os.path.join('data/OBIMD_HJ_diff/rubbings', img_path))
        H, W, C = image.shape     
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        
        masks = (masks * 255).astype(np.uint8)
        if masks.shape[0] == 1:
            masks = masks[None, ...]

        correct_result = dict(image_info=pred_result['image_info'], annotations=[])
        process_count = 0
        for ann in pred_result['annotations']:
            if ann['predicted_iou'] >= iou_threshold_last_round:
                correct_result['annotations'].append(ann)
                continue
            mask = np.asfortranarray(masks[process_count, 0])
            rle = maskUtils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle)
            ann.update({
                'segmentation': rle,
                'area': int(area),
                'bbox': bbox.astype(int).tolist(),
                'predicted_iou': float(scores[process_count]),
            })
            process_count += 1
            correct_result['annotations'].append(ann)
        with open(os.path.join(output_dir, file), 'w') as f:
            json.dump(correct_result, f, indent=2, ensure_ascii=False)