from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import json
import os
import random
from sam2.data_utils.utils import calculate_stability_score

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda:1')
predictor = SAM2ImagePredictor(sam2_model)

rubbing_dir = "data/OBIMD_raw_hj/rubbing"
facsimile_dir = "data/OBIMD_raw_hj/facsimile"
json_dir = "data/OBIMD_raw_hj/facsimile_json" # load gt bbox
vis_dir = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2.yaml/visualization" # save in ckpt dir

random.seed(42)
file_list = random.sample(os.listdir(rubbing_dir), 100) # 对齐
for img_path in sorted(file_list):
    image = cv2.imread(os.path.join(rubbing_dir, img_path))
    facsimile = cv2.imread(os.path.join(facsimile_dir, img_path))
    base_name = os.path.splitext(img_path)[0]
    with open(os.path.join(json_dir, f'{base_name}.json')) as f:
        data = json.load(f)
    predictor.set_image(image)
    input_box = []
    for d in data['annotations']:
        input_box.append(d['bbox'])
    input_box = np.array(input_box).reshape(-1, 4)
    input_box[:, 2:] += input_box[:, :2]  # Convert from [x, y, w, h] to [x0, y0, x1, y1]
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
        return_logits=True
    )
    stability_scores = calculate_stability_score(masks)
    masks = (masks > 0).any(axis=0)[0].astype(np.uint8) * 255 # H, W?
    for box, score, stab in zip(input_box, scores, stability_scores):
        x0, y0, x1, y1 = box
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        iou_text = f"IoU: {float(score):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(iou_text, font, font_scale, font_thickness)[0]
        text_x = int(x0 + ((x1-x0) - text_size[0]) / 2)
        text_y = int(y0 - 5)  # Position above the box
        cv2.putText(image, iou_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, img_path)
    # Apply a colormap to the facsimile for better visualization
    colored_facsimile = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, colored_facsimile, 0.5, 0)
    combined = np.hstack((overlay, facsimile))
    cv2.imwrite(output_path, combined)