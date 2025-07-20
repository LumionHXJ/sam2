# 自监督飞轮阶段，使用yolov12的检测结果作为box prompt，进行打标

from sam2.build_sam import build_sam2_video_predictor

import cv2
import numpy as np
import json
from tqdm import tqdm
import os
from pycocotools import mask as maskUtils
import torch
from sam2.data_utils.utils import load_raw_annotations, calculate_stability_score

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2_correction/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_dir = 'data/OBIMD_stage3_correction/rubbing_sav'
mask_threshold = 0
stability_score_offset = 1.0 
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
os.makedirs('data/OBIMD_stage3_correction/facsimile_json', exist_ok=True)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    char_id = 0
    # load train_list
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = [line.strip() for line in f.readlines()]

    # load raw data
    data_lookup = load_raw_annotations("data/OBIMD_raw_hj/label_filtered.json", "data/OBIMD_raw_hj/facsimile", ignore_null=True) # WARNING: NO null char

    for image_id, path in tqdm(enumerate(train_list)):
        facsimile = cv2.imread(os.path.join('data/OBIMD_stage3_correction/facsimile', path+'.jpg'), cv2.IMREAD_GRAYSCALE)
        H, W = facsimile.shape
        if os.path.exists(os.path.join('data/OBIMD_stage3_correction/facsimile_json', path+'.json')):
            continue
        input_box = []
        for char in data_lookup[path]:
            x, y, w, h = list(map(int, char['Position'].split(',')))
            input_box.append([x, y, x + w, y + h])
        inference_state = predictor.init_state(video_path=os.path.join(video_dir, path.split('.')[0]))
        predictor.reset_state(inference_state)
        input_box = np.array(input_box).reshape(-1, 4)
        
        for i in range(len(input_box)):
            # Add misalign facs to first frame as mask
            frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                mask=facsimile,
            )
        for i, box in enumerate(input_box):
            # Add box to the second frame
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=1,
                obj_id=i,
                box= box[None, :],
            )

        stability_scores = calculate_stability_score(out_mask_logits)
        masks = (out_mask_logits.cpu().numpy() > 0).astype(np.uint8) * 255
        scores = [d['cond_frame_outputs'][1]['ious'].cpu()[0] for d in inference_state['temp_output_dict_per_obj'].values()]

        pred_result = dict(image_info=dict(image_id=image_id, width=W, height=H, file_name=path+'.jpg'), annotations=[])
        for mask, score, stab_score, box in zip(masks, scores, stability_scores, input_box):
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
        with open(os.path.join('data/OBIMD_stage3_correction/facsimile_json', path+'.json'), 'w') as f:
            json.dump(pred_result, f, indent=2, ensure_ascii=False)