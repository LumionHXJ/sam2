import json
import os
import glob

def convert_sa1b_to_yolo(input_dir, output_dir, train_list, test_list):
    """
    将 SA1B 格式转换为 YOLO 格式
    :param input_dir: SA1B 格式的 JSON 文件目录
    :param output_dir: YOLO 格式的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/test'), exist_ok=True)
    
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    for json_file in json_files:
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像信息
        image_info = data.get('image_info', {})
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        file_name = image_info.get('file_name', '')
        
        # 生成 YOLO 文件名
        base_name = os.path.splitext(file_name)[0]
        
        # 处理标注
        annotations = data.get('annotations', [])
        yolo_lines = []
        
        for ann in annotations:
            # 获取 crop_box
            crop_box = ann.get('bbox', [])
            if len(crop_box) != 4:
                continue
            
            x, y, w, h = crop_box
            
            # 计算 YOLO 格式的中心坐标和归一化尺寸
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height
            
            # 类别 ID（默认为 0，可根据实际情况调整）
            class_id = 0
            
            # 生成 YOLO 格式的行
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)
        
        # 写入 YOLO 文件
        if base_name in train_list:
            with open(os.path.join(output_dir, 'labels/train', f'{base_name}.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
        elif base_name in test_list:
            with open(os.path.join(output_dir, 'labels/test', f'{base_name}.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
        else:
            raise ValueError(f"文件 {base_name} 不在训练集或测试集中")
        print(f"Converted {file_name} to YOLO format")


if __name__ == "__main__":
    input_directory = 'data/OBIMD_iou0.6/stage3/facsimile_json'
    output_directory = 'data/OBIMD_iou0.6/stage3/yolo'
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = [f.strip() for f in f.readlines()]
    with open("data/OBIMD_raw_hj/test.txt") as f:
        test_list = [f.strip() for f in f.readlines()]
    
    convert_sa1b_to_yolo(input_directory, output_directory, set(train_list), set(test_list))
    print("Conversion completed!")