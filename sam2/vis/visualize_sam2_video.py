from sam2.build_sam import build_sam2_video_predictor
import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import random

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2_correction/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
TEST100 = True

if TEST100:
    video_dir = 'data/OBIMD_test100/rubbing_sav/'
    image_dir = 'data/OBIMD_test100/JPEGImages'
    facsimile_dir = "data/OBIMD_raw_hj/facsimile"
    json_dir = "data/OBIMD_test100/facsimile_json" # load gt bbox
    vis_dir = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2_correction/visualize_test100" # save in ckpt dir
    gt_facsimile_dir = 'data/OBIMD_test100/VOC/'
else:
    video_dir = 'data/OBIMD_raw_hj/rubbing_sav'
    image_dir = 'data/OBIMD_raw_hj/rubbing'
    facsimile_dir = "data/OBIMD_raw_hj/facsimile"
    json_dir = "data/OBIMD_raw_hj/facsimile_json" # load gt bbox
    vis_dir = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2_correction/visualize" # save in ckpt dir
    gt_facsimile_dir = None

random.seed(42)
file_list = random.sample(os.listdir(video_dir), 100)
for path in file_list:
    img_path = path + '.jpg'
    image = cv2.imread(os.path.join(image_dir, img_path))
    inference_state = predictor.init_state(video_path=os.path.join(video_dir, img_path.split('.')[0]))
    predictor.reset_state(inference_state)
    facsimile = cv2.imread(f'{facsimile_dir}/{img_path}', cv2.IMREAD_GRAYSCALE)
    if gt_facsimile_dir is not None:
        gt_facsimile = cv2.imread(f'{gt_facsimile_dir}/{img_path}', cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(json_dir, path+'.json')) as f:
        data = json.load(f)
    input_box = []
    for i, d in enumerate(data['annotations']):
        # Add misalign facs to first frame as mask
        frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i,
            mask=facsimile,
        )
    for i, d in enumerate(data['annotations']):
        box = np.array(d['bbox']).reshape(-1, 4)
        box[:, 2:] += box[:, :2]  # Convert from [x, y, w, h] to [x0, y0, x1, y1]
        # Add box to the second frame
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=1,
            obj_id=i,
            box= box[None, :],
        )
        input_box.append(box[0])

    mask = out_mask_logits.cpu().numpy() > 0
    mask = mask.any(axis=0)[0].astype(np.uint8) * 255 # H, W?
    ious = [d['cond_frame_outputs'][1]['ious'].cpu()[0] for d in inference_state['temp_output_dict_per_obj'].values()]
    for box, score in zip(input_box, ious):
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
    colored_facsimile = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, colored_facsimile, 0.5, 0)
    if gt_facsimile_dir is not None:
        overlay_facs = cv2.addWeighted(gt_facsimile[..., np.newaxis].repeat(3, axis=2), 0.5, colored_facsimile, 0.5, 0)
    else:
        overlay_facs = cv2.addWeighted(facsimile[..., np.newaxis].repeat(3, axis=2), 0.5, colored_facsimile, 0.5, 0)
    combined = np.hstack((overlay, overlay_facs))
    cv2.imwrite(output_path, combined)