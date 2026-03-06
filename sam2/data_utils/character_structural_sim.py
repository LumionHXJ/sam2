# 
import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from multiprocessing import Pool, cpu_count

def calculate_metrics(args: tuple[str, str, int, tuple[int, int]]) -> tuple[float, float]:
    gt_path, pred_path, data_range, image_size = args

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, image_size, interpolation=cv2.INTER_LINEAR)
    pred = cv2.resize(pred, image_size, interpolation=cv2.INTER_LINEAR)
    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    
    psnr_score = psnr(gt, pred, data_range=data_range)
    ssim_score = ssim(
        gt, pred, 
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        multichannel=False
    )
    return psnr_score, ssim_score

def main():
    psnr_result, ssim_result = [], []
    exp_name = "obichariot"
    base_dir = f"data/OBIMD_cls/{exp_name}"
    data_range = 255
    image_size = (224, 224)

    tasks = []
    print(f"正在扫描目录: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            pred_path = os.path.join(root, file)
            gt_path = os.path.join(root.replace(exp_name, "test_hj"), file)
            tasks.append((gt_path, pred_path, data_range, image_size))
    
    print(f"发现 {len(tasks)} 个任务。")

    num_workers = cpu_count()
    print(f"使用 {num_workers} 个进程进行处理...")

    with Pool(processes=num_workers) as pool:
        # imap 是一个迭代器，它会在结果准备好时立即返回，适合与 tqdm 结合
        for psnr_score, ssim_score in tqdm(pool.imap(calculate_metrics, tasks), total=len(tasks)):
            # 过滤掉因文件问题导致的无效结果
            if not np.isnan(psnr_score) and not np.isnan(ssim_score):
                psnr_result.append(psnr_score)
                ssim_result.append(ssim_score)

    # 3. 计算并打印最终结果
    if psnr_result and ssim_result:
        avg_psnr = sum(psnr_result) / len(psnr_result)
        avg_ssim = sum(ssim_result) / len(ssim_result)
        print(f"\n处理完成！")
        print(f"平均 PSNR: {avg_psnr:.4f}")
        print(f"平均 SSIM: {avg_ssim:.4f}")
    else:
        print("\n所有任务都失败了，没有有效结果。")

if __name__ == "__main__":
    main()