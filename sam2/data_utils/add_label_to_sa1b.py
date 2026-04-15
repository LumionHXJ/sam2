import json
import os
from tqdm import tqdm

def load_label_map(label_json_path):
    """从 label.json 加载所有字符信息，建立 image_name -> crop_box -> info 的映射"""
    with open(label_json_path) as f:
        label_data = json.load(f)

    label_map = {}
    for data in label_data:
        image_name = os.path.basename(data['Facsimile']).split('.')[0]
        crop_box_to_info = {}

        for sentence in data.get('RecordUtilSentenceGroupVoList', []):
            for char in sentence.get('RecordUtilOracleCharVoList', []):
                # 将 Position 字符串 "x,y,w,h" 转换为列表 [x, y, w, h]
                crop_box = [int(x) for x in char['Position'].split(',')]
                crop_box_to_info[tuple(crop_box)] = {
                    'label': char.get('Label'),
                    'sublabel': char.get('SubLabel'),
                    'mark': char.get('Mark')
                }

        label_map[image_name] = crop_box_to_info
    return label_map

def add_label_to_sa1b_json(json_dir, label_json_path):
    """为 SA-1B 格式的 JSON 文件添加 Label"""
    label_map = load_label_map(label_json_path)

    matched_count = 0
    unmatched_count = 0
    total_annotations = 0
    total_files = 0

    for filename in tqdm(os.listdir(json_dir), desc="处理JSON文件"):
        if not filename.endswith('.json'):
            continue

        image_name = filename.split('.')[0]
        json_path = os.path.join(json_dir, filename)

        if image_name not in label_map:
            print(f"警告: {image_name} 不在 label.json 中")
            continue

        crop_box_to_info = label_map[image_name]

        with open(json_path) as f:
            data = json.load(f)

        # 添加 Label

        for annotation in data['annotations']:
            total_annotations += 1
            crop_box = annotation['crop_box']
            crop_box_tuple = tuple(crop_box)

            if crop_box_tuple in crop_box_to_info:
                info = crop_box_to_info[crop_box_tuple]
                annotation['Label'] = info['label']
                annotation['SubLabel'] = info['sublabel']
                annotation['Mark'] = info['mark']
                matched_count += 1
            else:
                annotation['Label'] = None
                annotation['SubLabel'] = None
                annotation['Mark'] = None
                unmatched_count += 1
                print(f"未匹配: {image_name} 的 crop_box {crop_box} 在 label.json 中未找到对应信息")

        # 保存更新后的 JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        total_files += 1

    print(f"总共处理 {total_files} 个文件")
    print(f"总共 {total_annotations} 个标注")
    print(f"匹配成功: {matched_count} 个")
    print(f"未匹配: {unmatched_count} 个")

if __name__ == "__main__":
    json_dir = "data/OBIMD_test100/facsimile_json"
    label_json_path = "data/OBIMD_raw_hj/label_proc.json"
    add_label_to_sa1b_json(json_dir, label_json_path)
