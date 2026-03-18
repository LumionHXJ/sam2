# 数据飞轮阶段间，使用当前最佳的模型，尝试修改之前的打标结果（如果置信度更高)

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
from sam2.data_utils.utils import calculate_stability_score

# checkpoint from last round
sam2_checkpoint = "sam2_logs/configs/sam2.1_training_stage1/sam2.1_hiera_l_OBIMD_coldstart.yaml/checkpoints/checkpoint_10.pt"
last_round_data = 'data/OBIMD_sam/coldstart/facsimile_json_full'
output_dir = 'data/OBIMD_sam/stage1/facsimile_json_full'
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l_highres.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:2')

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
            box=input_box,
            multimask_output=False,
            return_logits=True
        )
        
        if masks.shape[0] == 1:
            masks = masks[None]
        stability_scores = calculate_stability_score(masks)
        masks = (masks > 0).astype(np.uint8) * 255        

        correct_result = dict(image_info=pred_result['image_info'], annotations=[])
        for ann, mask, score, stab_score in zip(pred_result['annotations'], masks, scores, stability_scores):
            if ann['predicted_iou'] >= score:
                # 之前的结果置信度更高
                correct_result['annotations'].append(ann)
                continue
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle)
            ann.update({ # 例如crop box仍保留之前的值
                'segmentation': rle,
                'area': int(area),
                'bbox': bbox.astype(int).tolist(),
                'predicted_iou': float(score),
                'stability_score': float(stab_score),
            })
            correct_result['annotations'].append(ann)
        with open(os.path.join(output_dir, file), 'w') as f:
            json.dump(correct_result, f, indent=2, ensure_ascii=False)