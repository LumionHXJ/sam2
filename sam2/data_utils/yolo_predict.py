import multiprocessing
import os

import torch
from tqdm import tqdm
from ultralytics import YOLO

RUBBING_DIR = 'data/YQWY/rubbing'
OUTPUT_DIR = 'data/YQWY/yolo_prediction'
MODEL_PATH = "sam2_logs/yolo/260326-010345/weights/best.pt"


def task_worker(gpu_id, task_list):
    print(f"[GPU {gpu_id}] 进程启动，任务数量: {len(task_list)}")

    model = YOLO(MODEL_PATH, verbose=False)

    for img_name in tqdm(task_list, desc=f'[GPU {gpu_id}] 推理进度'):
        image_path = os.path.join(RUBBING_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + '.txt')

        # if os.path.exists(output_path):
        #     continue

        result = model(image_path, device=gpu_id)[0]
        result.save_txt(output_path)

    print(f"[GPU {gpu_id}] 进程完成所有任务。")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_processes = 4
    if num_processes == 0:
        raise RuntimeError('未检测到可用 GPU')

    image_list = sorted(
        img_name for img_name in os.listdir(RUBBING_DIR)
        if os.path.isfile(os.path.join(RUBBING_DIR, img_name))
    )
    total_tasks = len(image_list)
    print(f"总任务数: {total_tasks}")

    chunk_size = (total_tasks + num_processes - 1) // num_processes
    task_chunks = [image_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]

    processes = []
    for gpu_id in range(num_processes):
        p = multiprocessing.Process(target=task_worker, args=(gpu_id, task_chunks[gpu_id]))
        processes.append(p)
        p.start()
        print(f"进程 {gpu_id} (GPU {gpu_id}) 已启动，分配 {len(task_chunks[gpu_id])} 个任务。")

    for p in processes:
        p.join()

    print('所有进程已完成工作！')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
