import multiprocessing
import os
import shutil
from tqdm import tqdm
import json
import numpy as np
import cv2
import torch
from pycocotools import mask as maskUtils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.data_utils.utils import calculate_stability_score

SAM2_CHECKPOINT = "sam2_logs/configs/sam2.1_sam/sam2.1_hiera_l_OBIMD_stage2.yaml/checkpoints/checkpoint_10.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l_highres.yaml"
LAST_ROUND_DATA = 'data/OBIMD_sam/stage2/facsimile_json_full'
RUBBING_DIR = 'data/OBIMD_raw_hj/rubbing'
OUTPUT_DIR = 'data/OBIMD_sam/stage3/facsimile_json_full'

def task_worker(gpu_id, task_list):
    """
    每个GPU进程执行的任务函数。
    :param gpu_id: 分配给该进程的GPU ID (0-7)
    :param task_list: 该进程需要处理的文件列表
    """
    device = f'cuda:{gpu_id}'
    print(f"[GPU {gpu_id}] 进程启动，使用设备: {device}，任务数量: {len(task_list)}")

    try:
        sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor = SAM2ImagePredictor(sam2_model)

        for file in tqdm(task_list, desc=f'[GPU {gpu_id}] 处理进度'):
            if os.path.exists(os.path.join(OUTPUT_DIR, file)):
                continue
            with open(os.path.join(LAST_ROUND_DATA, file), 'r') as f:
                pred_result = json.load(f)
            input_box = []
            for ann in pred_result['annotations']:
                input_box.append(ann['crop_box'])
            if len(input_box) == 0:
                shutil.copy(os.path.join(LAST_ROUND_DATA, file), os.path.join(OUTPUT_DIR, file))
                continue
            input_box = np.array(input_box).reshape(-1, 4)
            input_box[:, 2:] += input_box[:, :2]

            img_path = file.replace('.json', '.jpg')
            image = cv2.imread(os.path.join(RUBBING_DIR, img_path))
            if image is None:
                print(f'[GPU {gpu_id}] 警告: 无法读取图片 {img_path}')
                continue
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
                    correct_result['annotations'].append(ann)
                    continue
                mask = np.asfortranarray(mask[0])
                rle = maskUtils.encode(mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                area = maskUtils.area(rle)
                bbox = maskUtils.toBbox(rle)
                ann.update({
                    'segmentation': rle,
                    'area': int(area),
                    'bbox': bbox.astype(int).tolist(),
                    'predicted_iou': float(score),
                    'stability_score': float(stab_score),
                })
                correct_result['annotations'].append(ann)
            with open(os.path.join(OUTPUT_DIR, file), 'w') as f:
                json.dump(correct_result, f, indent=2, ensure_ascii=False)

    print(f"[GPU {gpu_id}] 进程完成所有任务。")

def main():
    """主函数，负责启动多进程任务。"""
    num_processes = 4

    print("正在加载任务列表...")
    file_list = sorted(os.listdir(LAST_ROUND_DATA))
    total_tasks = len(file_list)
    print(f"总任务数: {total_tasks}")

    chunk_size = (total_tasks + num_processes - 1) // num_processes
    task_chunks = [file_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_processes)]

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
