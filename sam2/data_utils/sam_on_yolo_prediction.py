import os
import json
import cv2
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
import random
from pycocotools import mask as maskUtils
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.data_utils.utils import calculate_stability_score

# 配置参数
sam2_checkpoint = "sam2_logs/configs/sam2.1_iou_exp/iou0.6/sam2.1_hiera_l_OBIMD_stage4.yaml/checkpoints/checkpoint_5.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l_highres.yaml"
image_dir = 'data_YQWY/raw/rubbing'
yolo_pred_dir = 'data_YQWY/raw/yolo_prediction'
output_dir = 'data_YQWY/obichariot/facsimile_json'
bsz = 16

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def process_chunk(args):
    """处理数据切片的函数，每个进程执行此函数"""
    chunk, gpu_id = args
    
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化字符ID，确保每个进程的ID范围不重叠
    current_char_id = 0
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 在每个进程中单独初始化模型
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        for idx, img_path in enumerate(tqdm(chunk, desc=f"GPU {gpu_id} Processing")): 
            json_path = os.path.join(output_dir, f'{img_path.split(".")[0]}.json')
            if os.path.exists(json_path):
                continue
                
            yolo_txt = os.path.join(yolo_pred_dir, img_path.replace('.jpg', '.txt'))
            if not os.path.exists(yolo_txt):
                continue
                
            image_path = os.path.join(image_dir, img_path)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            H, W, _ = image.shape
            input_box = []
            
            with open(yolo_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    x, y, w, h = map(float, parts[1:5])
                    x0, y0 = ((x - w/2) * W), ((y - h/2) * H)
                    x1, y1 = ((x + w/2) * W), ((y + h/2) * H)
                    input_box.append([x0, y0, x1, y1])
            
            if not input_box:  # 没有有效的框则跳过
                continue
                
            # 进行预测
            predictor.set_image(image)
            input_box = np.array(input_box).reshape(-1, 4)

            masks, scores, _ = predictor.batch_predict(
                batchsize=bsz,
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
            pred_result = {
                'image_info': {
                    'image_id': idx,
                    'width': W,
                    'height': H,
                    'file_name': img_path
                },
                'annotations': []
            }
            for mask, score, box, stab_score in zip(masks, scores, input_box, stability_scores):
                mask = np.asfortranarray(mask[0])
                rle = maskUtils.encode(mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                area = maskUtils.area(rle)
                bbox = maskUtils.toBbox(rle)
                box[2:] = box[2:] - box[:2]
                
                pred_result['annotations'].append({
                    'id': current_char_id,
                    'bbox': bbox.astype(int).tolist(),
                    'area': int(area),
                    'segmentation': rle,
                    'predicted_iou': float(score),
                    'stability_score': float(stab_score),
                    'crop_box': box.astype(int).tolist()
                })
                current_char_id += 1
            with open(json_path, 'w') as f:
                json.dump(pred_result, f, indent=2, ensure_ascii=False)
            torch.cuda.empty_cache()
    
    return current_char_id

def split_data(data, num_chunks):
    chunk_size = max(1, len(data) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(data)
        chunks.append(data[start:end])
    return chunks

def main():
    all_images = sorted(os.listdir(image_dir))
    all_images = [img for img in all_images if not os.path.exists(os.path.join(output_dir, img.replace('.jpg', '.json'))) and os.path.exists(os.path.join(yolo_pred_dir, img.replace('.jpg', '.txt')))]
    random.seed(42)
    random.shuffle(all_images)
    
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}, 未处理图像数量：{len(all_images)}")
    data_chunks = split_data(all_images, num_gpus)
    
    process_args = [
        (data_chunks[i], i) 
        for i in range(num_gpus)
    ]
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        try:
            results = pool.map(process_chunk, process_args)
            print(f"所有进程完成。最终字符ID: {max(results) if results else 0}")
        except KeyboardInterrupt:
            print("检测到中断信号，正在终止所有进程...")
            pool.terminate()
        finally:
            pool.join()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
