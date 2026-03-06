from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import json
import os
import random
from sam2.data_utils.utils import calculate_stability_score

sam2_checkpoint = "sam2_logs/configs/sam2.1_iou_exp/iou0.6/sam2.1_hiera_l_OBIMD_stage4.yaml/checkpoints/checkpoint_5.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l_highres.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
predictor = SAM2ImagePredictor(sam2_model)

rubbing_dir = "data/OBIMD_test100/rubbing"
facsimile_dir = "data/OBIMD_test100/facsimile"
json_dir = "data/OBIMD_test100/facsimile_json" # load gt bbox
vis_dir = "sam2_logs/configs/sam2.1_iou_exp/iou0.6/sam2.1_hiera_l_OBIMD_stage4.yaml/visualization_gt" # save in ckpt dir

random.seed(42)
train_list = [f.split('.')[0] for f in os.listdir('data/OBIMD_test100/rubbing')]
file_list = random.sample(train_list, 100)
for path in sorted(file_list):
    img_path = path + '.jpg'
    image = cv2.imread(os.path.join(rubbing_dir, img_path))
    H, W, _ = image.shape
    facsimile = cv2.imread(os.path.join(facsimile_dir, img_path))
    try:
        with open(os.path.join(json_dir, f'{path}.json')) as f:
            data = json.load(f)
    except:
        print(f'{path}.json not exist')
        continue
    predictor.set_image(image)

    # 输入yolo结果
    # input_box = []
    # with open(os.path.join('data/OBIMD_test100/yolo_prediction', img_path.replace('.jpg', '.txt'))) as f:
    #     for line in f:
    #         box = list(map(float, line.strip().split()[1:]))
    #         x,y,w,h = box
    #         x0, y0 = (x - w/2) * W, (y - h/2) * H
    #         x1, y1 = (x + w/2) * W, (y + h/2) * H
    #         input_box.append([x0, y0, x1, y1])
    # input_box = np.array(input_box).reshape(-1, 4)
    
    # 输入ground truth
    input_box = []
    for d in data['annotations']:
        input_box.append(d['bbox'])
    input_box = np.array(input_box).reshape(-1, 4)
    input_box[:, 2:] += input_box[:, :2] # Convert from [x, y, w, h] to [x0, y0, x1, y1]

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
        return_logits=True
    )
    if len(masks) == 1:
        masks = masks[None]
    stability_scores = calculate_stability_score(masks)
    color_mask = np.zeros_like(image)
    for box, score, stab, mask in zip(input_box, scores, stability_scores, masks):
        x0, y0, x1, y1 = box
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        color = np.random.randint(0, 256, size=3).tolist()
        color_mask[mask[0] > 0] = color
        # iou_text = f"IoU: {float(score):.2f}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # font_thickness = 1
        # text_size = cv2.getTextSize(iou_text, font, font_scale, font_thickness)[0]
        # text_x = int(x0 + ((x1-x0) - text_size[0]) / 2)
        # text_y = int(y0 - 5)  # Position above the box
        # cv2.putText(image, iou_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, img_path)
    # Apply a colormap to the facsimile for better visualization
    overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
    # facsimile = cv2.addWeighted(facsimile, 0.5, colored_facsimile, 0.5, 0)
    # combined = np.hstack((overlay, facsimile))
    cv2.imwrite(output_path, overlay)