import multiprocessing
import os
from tqdm import tqdm
import json
import numpy as np
import cv2
import torch
from pycocotools import mask as maskUtils

# ----------------- 你项目中的自定义模块 -----------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.data_utils.utils import (load_raw_annotations, calculate_stability_score,
                                  get_tight_box, filter_small_connected_components)
from sam2.data_utils.coldstart.generate_align_data_by_shift import optimize_label_with_shifts

# 全局常量定义 (在主进程中定义，子进程会继承)
SAM2_CHECKPOINT = "sam2_logs/configs/sam2.1_training_stage1/sam2.1_hiera_l_OBIMD_stage1.yaml/checkpoints/checkpoint_10.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l_highres.yaml"
SAM2_FACS_CHECKPOINT = 'sam2_logs/configs/sam2.1_training_stage1/sam2.1_hiera_l_OBIMD_facs.yaml/checkpoints/checkpoint_5.pt'

RUBBING_DIR = 'data/OBIMD_raw_hj/rubbing'
FACS_DIR = 'data/OBIMD_raw_hj/facsimile'
ACCEPTED_DIR = 'data/OBIMD_iou0.6/stage2/facsimile_json'
OUTPUT_DIR = 'data/OBIMD_iou0.6/stage3/facsimile_json_replace'

def task_worker(gpu_id, task_list):
    """
    每个GPU进程执行的任务函数。
    :param gpu_id: 分配给该进程的GPU ID (0-7)
    :param task_list: 该进程需要处理的图片路径列表
    """
    device = f'cuda:{gpu_id}'
    print(f"[GPU {gpu_id}] 进程启动，使用设备: {device}，任务数量: {len(task_list)}")

    # 1. 在子进程中加载模型
    try:
        sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        sam2_facs_model = build_sam2(MODEL_CFG, SAM2_FACS_CHECKPOINT, device=device)
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return

    # 2. 加载原始标注数据
    data_lookup = load_raw_annotations("data/OBIMD_raw_hj/label_filt_train.json",
                                       "data/OBIMD_raw_hj/facsimile",
                                       ignore_null=True)

    # 3. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. 开始处理任务
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor = SAM2ImagePredictor(sam2_model)
        facs_predictor = SAM2ImagePredictor(sam2_facs_model)

        char_id = 0

        # 使用带前缀的tqdm进度条
        for path in tqdm(task_list, desc=f'[GPU {gpu_id}] 处理进度'):
            if os.path.exists(os.path.join(OUTPUT_DIR, path + '.json')):
                continue

            # ... [原代码中处理单张图片的逻辑保持不变] ...
            input_box = []
            ignore_box = []

            if os.path.exists(os.path.join(ACCEPTED_DIR, path + '.json')):
                with open(os.path.join(ACCEPTED_DIR, path + '.json')) as f:
                    accepted_data = json.load(f)
                    for char in accepted_data['annotations']:
                        ignore_box.append(','.join([str(x) for x in char['crop_box']]))

            if path not in data_lookup or len(data_lookup[path]) == 0:
                # print(f'[GPU {gpu_id}] {path} has no gt') # 减少进程间的打印干扰
                continue

            for char in data_lookup[path]:
                if char['Position'] in ignore_box:
                    continue
                x, y, w, h = list(map(int, char['Position'].split(',')))
                input_box.append([x, y, x + w, y + h])

            if len(input_box) == 0:
                # print(f'[GPU {gpu_id}] {path} box has all been accept')
                continue

            image = cv2.imread(os.path.join(RUBBING_DIR, path + '.jpg'))
            facsimile = cv2.imread(os.path.join(FACS_DIR, path + '.jpg'))
            if image is None or facsimile is None:
                print(f'[GPU {gpu_id}] 警告: 无法读取图片 {path}')
                continue

            H, W, C = image.shape
            predictor.set_image(image)
            facs_predictor.set_image(facsimile)

            input_box = np.array(input_box).reshape(-1, 4)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
                return_logits=True
            )
            masks_facs, scores_facs, _ = facs_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
                return_logits=False
            )

            if len(masks) == 1:
                masks = masks[None]
            if len(masks_facs) == 1:
                masks_facs = masks_facs[None]

            stability_scores = calculate_stability_score(masks)
            masks = (masks > 0).astype(np.uint8) * 255
            masks_facs = (masks_facs > 0).astype(np.uint8) * 255

            pred_result = dict(image_info=dict(image_id=os.path.basename(path).split('.')[0],  # 使用文件名作为ID更可靠
                                               width=W, height=H, file_name=path + '.jpg'),
                               annotations=[])

            for mask, mask_fac, score, stab_score, box in zip(masks, masks_facs, scores, stability_scores, input_box):
                mask = filter_small_connected_components(mask[0])
                mask_fac = filter_small_connected_components(mask_fac[0])

                if not mask.any() or not mask_fac.any():
                    continue

                x1, y1, w1, h1 = get_tight_box(mask)
                x2, y2, w2, h2 = get_tight_box(mask_fac)

                mask, coverage = optimize_label_with_shifts(mask, mask_fac, [x2, y2, w2, h2], max_shift=20)

                rle = maskUtils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                area = maskUtils.area(rle)
                bbox = maskUtils.toBbox(rle)

                box[2:] = box[2:] - box[:2]  # Convert from [x0, y0, x1, y1] to [x, y, w, h]
                crop_box = box.astype(int).tolist()

                pred_result['annotations'].append(dict(id=char_id,
                                                       bbox=bbox.astype(int).tolist(),
                                                       area=int(area),
                                                       segmentation=rle,
                                                       predicted_iou=float(coverage),
                                                       stability_score=float(stab_score),
                                                       crop_box=crop_box))
                char_id += 1

            with open(os.path.join(OUTPUT_DIR, path + '.json'), 'w') as f:
                json.dump(pred_result, f, indent=2, ensure_ascii=False)

    print(f"[GPU {gpu_id}] 进程完成所有任务。")

def main():
    """主函数，负责启动多进程任务。"""
    num_processes = 2  # 使用8个进程

    # 1. 加载完整的任务列表
    print("正在加载任务列表...")
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = [line.strip() for line in f.readlines()]

    total_tasks = len(train_list)
    print(f"总任务数: {total_tasks}")

    # 2. 分割任务列表
    chunk_size = (total_tasks + num_processes - 1) // num_processes  # 向上取整
    task_chunks = [train_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_processes)]

    # 3. 创建并启动进程
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=task_worker, args=(i, task_chunks[i]))
        processes.append(p)
        p.start()
        print(f"进程 {i} (GPU {i}) 已启动，分配 {len(task_chunks[i])} 个任务。")

    for p in processes:
        p.join()

    print("\n所有进程已完成工作！")

if __name__ == "__main__":
    main()