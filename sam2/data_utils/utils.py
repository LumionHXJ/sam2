import numpy as np
import json
from tqdm import tqdm
import os
import cv2
from PIL import Image
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.nn.functional as F
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_tight_box(mask):
    """return in x,y,w,h format, +1 for slice op"""
    assert mask.dtype == np.uint8
    mask = mask > 127
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax - cmin + 1, rmax - rmin + 1]

def get_tight_box_torch(mask):
    """return in x,y,x,y format, +1 for slice op"""
    rows = mask.any(dim=1)
    cols = mask.any(dim=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax + 1, rmax + 1]

def get_tight_mask(mask):
    x,y,w,h = get_tight_box(mask)
    return mask[y:y+h, x:x+w]

def rotate_image(image, angle):
    """rotate image by angle, image center serve as a pivot"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    angle_rad = np.radians(angle)
    new_w = int((h * np.abs(np.sin(angle_rad))) + (w * np.abs(np.cos(angle_rad))))
    new_h = int((h * np.abs(np.cos(angle_rad))) + (w * np.abs(np.sin(angle_rad))))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv2.warpAffine(
        image, 
        M, 
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 
    )
    return rotated_image

def calculate_stability_score(
    masks, mask_threshold: float = 0, threshold_offset: float = 1
):
    masks = masks.reshape(masks.shape[0], -1)
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1)
    )
    return intersections / unions

def calc_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def calc_box_isin(box1, box2):
    """input xyxy format, return True if box1 is in box2"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4

def calc_mask_coverage(mask1, mask2):
    """calculate how many pixels in mask1 are covered by mask2"""
    assert mask1.dtype == np.uint8 and mask2.dtype == np.uint8
    return np.sum(mask1[mask2 > 127]/255) / np.sum(mask1/255)

def calc_mask_iou(mask1, mask2):
    assert mask1.dtype == np.uint8 and mask2.dtype == np.uint8
    mask1 = mask1 > 127
    mask2 = mask2 > 127
    return np.sum(mask1[mask2]) / (np.sum(mask1) + np.sum(mask2) - np.sum(mask1[mask2]))

def load_raw_annotations(data_path, segmap_root, ignore_null=False):
    with open(data_path) as f:
        datas = json.load(f)
    data_lookup = dict()
    for image_id, data in tqdm(enumerate(datas)):
        image_name = os.path.basename(data['Rubbing'])
        if not os.path.exists(os.path.join(segmap_root, image_name)):
            continue
        W, H = Image.open(os.path.join(segmap_root, image_name)).size
        # CHECK SHAPE etc.
        oracle_chars = []
        for sentence in data['RecordUtilSentenceGroupVoList']:
            for char in sentence["RecordUtilOracleCharVoList"]:
                if ignore_null and char['Label'] is None:
                    continue
                x, y, w, h = list(map(int, char['Position'].split(',')))
                if x < 0 or y < 0 or x + w > W or y + h > H:
                    continue
                oracle_chars.append(char)
        data_lookup[image_name.split('.')[0]] = oracle_chars
    return data_lookup

def load_null_annotations(data_path):
    with open(data_path) as f:
        datas = json.load(f)
    data_lookup = dict()
    for image_id, data in enumerate(datas):
        image_name = os.path.basename(data['Rubbing'])
        # CHECK SHAPE etc.
        oracle_chars = []
        for sentence in data['RecordUtilSentenceGroupVoList']:
            for char in sentence["RecordUtilOracleCharVoList"]:
                if char['Label'] is None:
                    oracle_chars.append(char) 
        data_lookup[image_name.split('.')[0]] = oracle_chars
    return data_lookup

def filter_box_by_iou(box_to_filter, ignore_boxes, thr=0.5):
    """如果和ignore_boxes的box有iou大于thr的，则返回True"""
    for ignore in ignore_boxes:
        if calc_box_iou(box_to_filter, ignore) > thr:
            return True
    return False

def calculate_mask_ap(gt_data, pred_data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as gt_file, \
         tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as pred_file:

        json.dump(gt_data, gt_file)
        json.dump(pred_data, pred_file)
        gt_path = gt_file.name
        pred_path = pred_file.name
    
    try:
        coco_gt = COCO(gt_path)
        coco_pred = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        
        return {
            'mask_ap': stats[0],         # Mask AP @[0.5:0.95]
            'mask_ap50': stats[1],       # Mask AP @0.5
            'mask_ap75': stats[2],       # Mask AP @0.75
            'mask_ap_small': stats[3],   # 小目标Mask AP
            'mask_ap_medium': stats[4],  # 中目标Mask AP
            'mask_ap_large': stats[5]    # 大目标Mask AP
        }
        
    finally:
        os.remove(gt_path)
        os.remove(pred_path)

def extract_connected_component_opencv(mask, start_x, start_y, connectivity=4):
    """
    使用OpenCV从指定点提取连通分支
    
    参数:
        mask: 二值图像掩码，numpy数组，前景为255，背景为0（OpenCV默认格式）
        connectivity: 连通性，4或8，默认为4
        
    返回:
        component: 包含连通分支所有坐标的列表，格式为[(x1,y1), (x2,y2), ...]
        component_mask: 与输入mask同尺寸的二值掩码，连通分支区域为255，其余为0
    """
    if mask[start_y, start_x] != 255:
        raise ValueError("起始点不是前景像素（掩码值不为255）")  
    mask_copy = mask.copy()
    
    fill_value = 128 
    
    _, filled_mask, _, _ = cv2.floodFill(
        mask_copy, 
        None, 
        (start_x, start_y),  # OpenCV的坐标是(y, x)即(col, row)
        fill_value, 
        loDiff=0, 
        upDiff=0, 
        flags=connectivity | cv2.FLOODFILL_FIXED_RANGE
    )
    
    component_mask = np.zeros_like(mask)
    component_mask[filled_mask == fill_value] = 255
    component = np.argwhere(component_mask == 255)
    component = [tuple(coord) for coord in component]
    
    return component, component_mask

def sliding_window_inference(predictor: SAM2ImagePredictor, image: np.array, box, window_size=1024):
    """滑动窗口推理，首先将短边resize
    box: 输入的box，格式为(x1, y1, x2, y2)，形状为N, 4
    """
    h, w = image.shape[:2]
    n = box.shape[0]
    scores = np.zeros((n, 1))
    processed = np.zeros((n, ), dtype=bool)

    if w <= h:
        scale = max(window_size / w, 1)
    else:
        scale = max(window_size / h, 1)
    new_w = max(int(w * scale), 1024)
    new_h = max(int(h * scale), 1024)
    image = cv2.resize(image, (new_w, new_h))
    scaled_boxes = box.copy().astype(np.float64)
    scaled_boxes[:, [0, 2]] *= scale 
    scaled_boxes[:, [1, 3]] *= scale 
    scaled_boxes = scaled_boxes.astype(np.int64)
    masks = np.zeros((n, 1, new_h, new_w))

    stride_w = (scaled_boxes[:, 2] - scaled_boxes[:, 0]).min() - 1
    stride_h = (scaled_boxes[:, 3] - scaled_boxes[:, 1]).min() - 1
    for i in list(range(0, new_h - window_size + 1, stride_h)) + [new_h - window_size]:
        for j in list(range(0, new_w - window_size + 1, stride_w)) + [new_w - window_size]:
            # 提取所有在窗口内的box
            curr_proc = np.zeros_like(processed)
            for idx in range(n):
                if not processed[idx] and calc_box_isin(scaled_boxes[idx], (j, i, j+window_size, i+window_size)):
                    curr_proc[idx] = True
            if not curr_proc.any():
                continue
            input_box = scaled_boxes[curr_proc]
            input_box[:, [0, 2]] -= j
            input_box[:, [1, 3]] -= i

            window = image[i:i+window_size, j:j+window_size]
            predictor.set_image(window)
            curr_masks, curr_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None],
                multimask_output=False,
            )
            if len(curr_masks) == 1:
                curr_masks = curr_masks[None]
            masks[curr_proc, :, i:i+window_size, j:j+window_size] = curr_masks
            scores[curr_proc] = curr_scores
            processed[curr_proc] = True
            if processed.all():
                masks = F.interpolate(torch.tensor(masks), (h, w)).numpy()
                return masks, scores
    for idx in range(n):
        if not processed[idx]:
            x0, y0, x1, y1 = scaled_boxes[idx]
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            w_ = max(w * 1.2, 1024)
            h_ = max(h * 1.2, 1024)
            x0_, y0_ = int(max(x- w_/2, 0)), int(max(y - h_/2, 0))
            x1_, y1_ = int(min(x + w_/2, new_w)), int(min(y + h_/2, new_h))
            crop_image = image[y0_:y1_, x0_:x1_]
            crop_box = np.array([x0-x0_, y0-y0_, x1-x0_, y1-y0_]).reshape(1, 1, 4)

            predictor.set_image(crop_image)
            mask, score, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=crop_box,
                multimask_output=False,
            )
            masks[idx, :, y0_:y1_, x0_:x1_] = mask[0]
            scores[idx] = score
    masks = F.interpolate(torch.tensor(masks), (h, w)).numpy()
    return masks, scores

def filter_small_connected_components(binary_img, min_area=20):
    """
    过滤二值图像中面积小于指定阈值的连通分支
    
    参数:
        binary_img: 输入的二值图像 (单通道，0表示背景，255表示前景)
        min_area: 最小保留面积，默认20
        
    返回:
        filtered_img: 过滤后的二值图像
    """
    assert len(binary_img.shape) == 2 and binary_img.dtype == np.uint8
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8) 
    
    filtered_img = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_img[labels == i] = 255
    return filtered_img

def update_training_dataset(
    input_dir: str, 
    output_dir: str, 
    accept_dir: str = None, 
    train_list: list = [], 
    new_train_txt: str = '',
    iou_threshold: float = 0.6):
    """
    更新训练数据集，将input_dir中的文件根据train_list中的文件名复制到output_dir中，
    同时将accept_dir中的文件也复制到output_dir中。
    """
    os.makedirs(output_dir, exist_ok=True)
    total_annotations = 0
    for file in train_list:
        file = file + '.json'
        annotations = []
        image_info = {}
        if accept_dir is not None and os.path.exists(os.path.join(accept_dir, file)):
            with open(os.path.join(accept_dir, file)) as f:
                data = json.load(f)
            annotations.extend(data['annotations'])
            image_info = data['image_info']

        if os.path.exists(os.path.join(input_dir, file)):
            try:
                with open(os.path.join(input_dir, file)) as f:
                    data = json.load(f)
            except:
                raise ValueError(f"Error loading {file} in {input_dir}")
            annotations.extend([annotation for annotation in data['annotations'] if annotation['predicted_iou'] > iou_threshold])
            image_info = data['image_info']
            
        filter_annotions = dict(image_info=image_info, annotations=annotations)
        total_annotations += len(filter_annotions['annotations'])
        if len(filter_annotions['annotations']) == 0:
            continue
        with open(os.path.join(output_dir, file), 'w') as f:
            json.dump(filter_annotions, f, indent=2)

    print(f"Total accept annotations: {total_annotations}")
    with open(new_train_txt, 'w') as f:
        for file in os.listdir(output_dir):
            if file.split('.')[0] in train_list:
                f.write(file.split('.')[0] + '\n')


def filter_training_dataset(
    input_dir: str, 
    output_dir: str, 
    train_list: list = [], 
    new_train_txt: str = '',
    iou_threshold: float = 0.6):
    """
    筛选训练数据集，只保留 input_dir 中 predicted_iou 大于阈值的标注，输出到 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)
    total_annotations = 0
    for file in train_list:
        file = file + '.json'
        annotations = []
        image_info = {}
        if os.path.exists(os.path.join(input_dir, file)):
            try:
                with open(os.path.join(input_dir, file)) as f:
                    data = json.load(f)
            except:
                raise ValueError(f"Error loading {file} in {input_dir}")
            annotations = [annotation for annotation in data['annotations'] if annotation['predicted_iou'] > iou_threshold]
            image_info = data['image_info']
            
        filter_annotions = dict(image_info=image_info, annotations=annotations)
        total_annotations += len(filter_annotions['annotations'])
        if len(filter_annotions['annotations']) == 0:
            continue
        with open(os.path.join(output_dir, file), 'w') as f:
            json.dump(filter_annotions, f, indent=2)

    print(f"Total accept annotations: {total_annotations}")
    with open(new_train_txt, 'w') as f:
        for file in os.listdir(output_dir):
            if file.split('.')[0] in train_list:
                f.write(file.split('.')[0] + '\n')