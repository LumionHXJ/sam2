# 过滤原始标注中的空框
import json
import os
from tqdm import tqdm
import cv2
import numpy as np

def check_subset(rub, facs):
    intersection = np.logical_and(rub, facs).sum()
    return intersection / facs.sum() if facs.sum() != 0 else 1.0
    
def check_borderline(rubbing, facsimile, bbox, max_shift=10):
    """检查没有类别标注的字符框是否在borderline上"""
    x, y, w, h = bbox
    cropped_facs = facsimile[y:y+h, x:x+w]
    if cropped_facs.sum() == 0:
        return True
    for dx in range(-max_shift, max_shift + 1, 2):
        for dy in range(-max_shift, max_shift + 1, 2):
            if x+dx < 0 or y+dy < 0 or x+dx+w > rubbing.shape[1] or y+dy+h > rubbing.shape[0]:
                continue # 越界
            cropped_rub = rubbing[y+dy:y+dy+h, x+dx:x+dx+w]
            if check_subset(cropped_rub > 0, cropped_facs > 0) > 0.99:
                return True
    return False

def filter_raw_annotations(data_path, rubbing_root, segmap_root, output_path):
    """处理完整的初始标注，清洗空框
    segmap_root使用完整的无边线版本
    """
    with open(data_path) as f:
        datas = json.load(f)
    new_datas = []
    for image_id, data in tqdm(enumerate(datas)):
        new_data = dict(Facsimile=data['Facsimile'], Rubbing=data['Rubbing'], RubbingName=data['RubbingName'], RecordUtilSentenceGroupVoList=[])
        image_name = os.path.basename(data['Rubbing'])

        if not os.path.exists(os.path.join(segmap_root, image_name)):
            print(f"Facsimile image {image_name} not found in {segmap_root}")
            continue
        
        rubbing = cv2.imread(os.path.join(rubbing_root, image_name), cv2.IMREAD_GRAYSCALE)
        facsimile = cv2.imread(os.path.join(segmap_root, image_name), flags=cv2.IMREAD_GRAYSCALE)
        _, binary_rubbing = cv2.threshold(rubbing, 127, 255, cv2.THRESH_BINARY)
        H, W = rubbing.shape[:2]

        # CHECK SHAPE etc.
        for sentence in data['RecordUtilSentenceGroupVoList']:
            new_sentence = dict(GroupCategory=sentence['GroupCategory'], RecordUtilOracleCharVoList=[])
            for char in sentence["RecordUtilOracleCharVoList"]:
                x, y, w, h = list(map(int, char['Position'].split(',')))
                # 训练数据排除原则：越界 + 空类别 + 边界/大噪声
                # 测试数据排除原则：越界
                if x < 0 or y < 0 or x + w > W or y + h > H: # 越界
                    continue
                if char['Label'] is None:
                    continue
                else:
                    if not check_borderline(binary_rubbing, facsimile, list(map(int, char['Position'].split(',')))):
                        new_sentence['RecordUtilOracleCharVoList'].append(char)
            new_data['RecordUtilSentenceGroupVoList'].append(new_sentence)
        new_datas.append(new_data)
    with open(output_path, 'w') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    filter_raw_annotations(
        data_path='data/OBIMD_raw_hj/label.json',
        rubbing_root='data/OBIMD_raw_hj/rubbing',
        segmap_root='data/OBIMD_raw_hj/facsimile',
        output_path='data/OBIMD_raw_hj/label_filt_train.json'
    )