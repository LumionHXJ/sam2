import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from pycocotools import mask as maskUtils

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def optimize_label_with_shifts(rubbing, facsimile, bbox, max_shift=10):
    """接受：完整的二值化拓片+字符部分+box"""
    x, y, w, h = bbox
    cropped_facs = facsimile[y:y+h, x:x+w]

    best_iou = -1
    best_shifted = (0, 0)

    for dx in range(-max_shift, max_shift + 1, 2):
        for dy in range(-max_shift, max_shift + 1, 2):
            if x+dx < 0 or y+dy < 0 or x+dx+w > rubbing.shape[1] or y+dy+h > rubbing.shape[0]:
                continue # 越界
            cropped_rub = rubbing[y+dy:y+dy+h, x+dx:x+dx+w]

            iou = calculate_iou(cropped_rub > 0, cropped_facs > 0)
            if iou > best_iou:
                best_iou = iou
                best_shifted = (dx, dy)

    refine_facsimile = np.zeros_like(facsimile, dtype=np.uint8)
    refine_facsimile[y+best_shifted[1]:y+best_shifted[1]+h, x+best_shifted[0]:x+best_shifted[0]+w] = cropped_facs
    return refine_facsimile, best_iou

def generate_align_data_by_shift(
    train_list,
    rubbing_dir,
    sa1b_json_dir, # 第一步中，需要使用没有边线的数据，这样预测才能够得到没有去线的版本
    output_dir
):
    for file in tqdm(
        sorted(train_list)
    ):
        os.makedirs(output_dir, exist_ok=True)

        rubbing_path = os.path.join(rubbing_dir, file+'.jpg')
        annotation_path = os.path.join(sa1b_json_dir, file+'.json')

        if not os.path.exists(rubbing_path):
            raise

        rubbing = cv2.imread(rubbing_path, cv2.IMREAD_GRAYSCALE)
        _, binary_rubbing = cv2.threshold(rubbing, 127, 255, cv2.THRESH_BINARY)

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        align_annotation = dict(image_info=annotation['image_info'], annotations=[])
        for ann in annotation['annotations']:
            segmentation = maskUtils.decode(ann['segmentation'])
            segmentation, best_iou = optimize_label_with_shifts(binary_rubbing, segmentation, ann['bbox'])
            rle = maskUtils.encode(np.asfortranarray(segmentation))
            rle['counts'] = rle['counts'].decode('utf-8')
            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle)
            align_annotation['annotations'].append({
                'segmentation': rle,
                'area': int(area),
                'bbox': bbox.astype(int).tolist(),
                'predicted_iou': float(best_iou),
            })                

        output_file = os.path.join(output_dir, file+'.json')

        with open(output_file, 'w') as f:
            json.dump(align_annotation, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = [l.strip() for l in f.readlines()]
    generate_align_data_by_shift(
        train_list=train_list,
        rubbing_dir='data/OBIMD_raw_hj/rubbing',
        sa1b_json_dir='data/OBIMD_raw_hj/facsimile_json_no_border',
        output_dir='data/OBIMD_stage1/facsimile_json_new'
    )
