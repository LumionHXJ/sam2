from sam2.build_sam import build_sam2_video_predictor
import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import random

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda:1')

video_dir = 'data/OBIMD_test100/rubbing_sav/'
image_dir = 'data/OBIMD_test100/JPEGImages'
facsimile_dir = "data/OBIMD_raw_hj/facsimile"
json_dir = "data/OBIMD_test100/facsimile_json" # load gt bbox
vis_dir = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_l_OBIMD_stage2.yaml/visualization" # save in ckpt dir

random.seed(42)
file_list = random.sample(os.listdir(video_dir), 10)
for path in file_list:
    img_path = path + '.jpg'
    image = cv2.imread(os.path.join(image_dir, img_path))
    inference_state = predictor.init_state(video_path=os.path.join(video_dir, img_path.split('.')[0]))
    predictor.reset_state(inference_state)
    facsimile = cv2.imread(f'{facsimile_dir}/{img_path}', cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(json_dir, path+'.json')) as f:
        data = json.load(f)
    input_box = []
    for i, d in enumerate(data['annotations']):
        box = np.array(d['bbox']).reshape(-1, 4)
        box[:, 2:] += box[:, :2]  # Convert from [x, y, w, h] to [x0, y0, x1, y1]
        # Add misalign facs to first frame as mask
        frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i,
            mask=facsimile,
        )
        # Add box to the second frame
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=1,
            obj_id=i,
            box= box[None, :],
        )
        input_box.append(box[0])
    
    video_segments = {}  
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    masks = np.array([mask for mask in video_segments[1].values()])
    masks = masks.any(axis=0)[0].astype(np.uint8) * 255 # H, W?
    for box in input_box:
        x0, y0, x1, y1 = box
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, img_path)
    # Apply a colormap to the facsimile for better visualization
    colored_facsimile = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, colored_facsimile, 0.5, 0)
    combined = np.hstack((overlay, facsimile[..., np.newaxis].repeat(3, axis=2)))
    cv2.imwrite(output_path, combined)