import json
import cv2
import numpy as np
import os
from tqdm import tqdm

def expand_box(bbox, expand_ratio=0.1, max_width=3000, max_height=3000):
    """扩张 box"""
    x, y, w, h = bbox
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)

    new_x = max(0, x - expand_w)
    new_y = max(0, y - expand_h)
    new_w = min(max_width - new_x, w + 2 * expand_w)
    new_h = min(max_height - new_y, h + 2 * expand_h)

    return [new_x, new_y, new_w, new_h]

def boxes_intersect(box1, box2):
    """检查两个 box 是否相交"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
    left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

    return not (right1 <= left2 or left1 >= right2 or bottom1 <= top2 or top1 >= bottom2)

def merge_boxes(boxes):
    """合并相交的 boxes"""
    if not boxes:
        return []

    merged = [boxes[0]]

    for box in boxes[1:]:
        merged_box = box
        i = 0
        while i < len(merged):
            if boxes_intersect(merged_box, merged[i]):
                # 合并这两个 box
                x1, y1, w1, h1 = merged_box
                x2, y2, w2, h2 = merged[i]

                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)

                merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                merged.pop(i)
            else:
                i += 1

        merged.append(merged_box)

    return merged

def point_in_box(px, py, box):
    """检查点是否在box内"""
    x, y, w, h = box
    return x <= px and x + w >= px and y <= py and y + h >= py

def main():
    # 路径配置
    label_json_path = "data/OBIMD_raw_hj/label_filtered.json"
    facsimile_dir = "data/OBIMD_raw_hd/facsimile"
    rubbing_dir = "data/OBIMD_raw_hd/rubbing"

    output_facsimile_dir = "data/OBIMD_test_hd_new/facsimile"
    output_rubbing_dir = "data/OBIMD_test_hd_new/rubbing"
    output_label_json_path = "data/OBIMD_test_hd_new/label.json"

    # 创建输出目录
    os.makedirs(output_facsimile_dir, exist_ok=True)
    os.makedirs(output_rubbing_dir, exist_ok=True)

    # 加载 label_filtered.json
    print("加载 label_filtered.json...")
    with open(label_json_path) as f:
        label_data = json.load(f)

    # 筛选 HD 数据
    hd_data = [data for data in label_data if data['RubbingName'].startswith("HD")]
    print(f"找到 {len(hd_data)} 个 HD 数据")

    total_split_count = 0
    file_count = 0
    total_inscription_boxes = 0
    total_merged_chars = 0
    label_list = []

    for data in tqdm(hd_data, desc="处理文件"):
        rubbing_name = data['RubbingName']
        facsimile_path = f"{facsimile_dir}/{rubbing_name.lower()}.jpg"
        rubbing_path = f"{rubbing_dir}/{rubbing_name.lower()}.jpg"

        # 读取图像
        facsimile_image = cv2.imread(facsimile_path)
        rubbing_image = cv2.imread(rubbing_path)

        if facsimile_image is None:
            print(f"无法读取 facsimile 图像: {facsimile_path}")
            continue
        if rubbing_image is None:
            print(f"无法读取 rubbing 图像: {rubbing_path}")
            continue

        img_h, img_w = facsimile_image.shape[:2]

        # 收集所有 InscriptionSentence 的 box 及其字符
        inscription_boxes = []  # 存储 bbox
        inscription_chars = []  # 存储对应的字符信息（原始数据）

        for sentence in data['RecordUtilSentenceGroupVoList']:
            if sentence['GroupCategory'].startswith("InscriptionSentence"):
                # 获取该 sentence 中所有字符的信息
                char_infos = []
                for char in sentence['RecordUtilOracleCharVoList']:
                    if char['Label'] is not None:
                        pos = [int(x) for x in char['Position'].split(',')]
                        char_infos.append({
                            'char': char,
                            'pos': pos,
                            'sentence_category': sentence['GroupCategory']
                        })

                if len(char_infos) <= 2:
                    continue

                # 计算该 sentence 的 bounding box
                char_positions = [info['pos'] for info in char_infos]
                chars_arr = np.array(char_positions)
                x_min = chars_arr[:, 0].min()
                y_min = chars_arr[:, 1].min()
                x_max = chars_arr[:, 0].max() + chars_arr[:, 2].max()
                y_max = chars_arr[:, 1].max() + chars_arr[:, 3].max()

                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                inscription_boxes.append(bbox)
                inscription_chars.append(char_infos)

        total_inscription_boxes += len(inscription_boxes)

        # 处理非 InscriptionSentence 的字符
        for sentence in data['RecordUtilSentenceGroupVoList']:
            if not sentence['GroupCategory'].startswith("InscriptionSentence"):
                for char in sentence['RecordUtilOracleCharVoList']:
                    if char['Label'] is None or char['Label'] in ['jrzjjh3g1r', '8gxzzbv7w8', 'p77d58vew4']:
                        continue
                    pos = [int(x) for x in char['Position'].split(',')]

                    # 检查该字符是否与任何 inscription_box 有交集
                    for i, box in enumerate(inscription_boxes):
                        if boxes_intersect(box, pos):
                            # 加入到对应的 box 中
                            inscription_chars[i].append({
                                'char': char,
                                'pos': pos,
                                'sentence_category': sentence['GroupCategory']
                            })
                            total_merged_chars += 1
                            break

        # 重新计算 inscription_boxes（加入新字符后）
        inscription_boxes = []
        for char_infos in inscription_chars:
            if char_infos:
                char_positions = [info['pos'] for info in char_infos]
                chars_arr = np.array(char_positions)
                x_min = chars_arr[:, 0].min()
                y_min = chars_arr[:, 1].min()
                x_max = chars_arr[:, 0].max() + chars_arr[:, 2].max()
                y_max = chars_arr[:, 1].max() + chars_arr[:, 3].max()

                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                inscription_boxes.append(bbox)

        # 扩张 box
        expanded_boxes = [expand_box(bbox, expand_ratio=0.1, max_width=img_w, max_height=img_h)
                          for bbox in inscription_boxes]

        # 合并有交集的 box
        merged_boxes = merge_boxes(expanded_boxes)

        # 处理每个 merged box
        for i, bbox in enumerate(merged_boxes):
            x, y, w, h = [int(v) for v in bbox]

            # 确保边界不越界
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w <= 0 or h <= 0:
                continue

            # 裁剪图像
            facsimile_crop = facsimile_image[y:y+h, x:x+w]
            rubbing_crop = rubbing_image[y:y+h, x:x+w]

            # 保存图像
            output_name = f"{rubbing_name.lower()}_{i}.jpg"
            facsimile_output_path = os.path.join(output_facsimile_dir, output_name)
            rubbing_output_path = os.path.join(output_rubbing_dir, output_name)

            cv2.imwrite(facsimile_output_path, facsimile_crop)
            cv2.imwrite(rubbing_output_path, rubbing_crop)

            # 收集该box内的字符信息，生成label.json条目
            char_list = []
            sentence_groups = {}

            for char_infos in inscription_chars:
                for char_info in char_infos:
                    char_pos = char_info['pos']
                    cx, cy, cw, ch = char_pos

                    # 检查字符中心点是否在box内
                    if point_in_box(cx + cw/2, cy + ch/2, bbox):
                        # 转换为相对坐标
                        rel_x = cx - x
                        rel_y = cy - y
                        rel_position = f"{int(rel_x)},{int(rel_y)},{cw},{ch}"

                        # 按sentence category分组
                        category = char_info['sentence_category']
                        if category not in sentence_groups:
                            sentence_groups[category] = []
                        sentence_groups[category].append({
                            'Position': rel_position,
                            'OrderNumber': char_info['char']['OrderNumber'],
                            'SeatFont': char_info['char']['SeatFont'],
                            'Mark': char_info['char']['Mark'],
                            'Label': char_info['char']['Label'],
                            'SubLabel': char_info['char']['SubLabel']
                        })

            # 按OrderNumber排序
            for category in sentence_groups:
                sentence_groups[category].sort(key=lambda x: x['OrderNumber'])

            # 构建RecordUtilSentenceGroupVoList
            record_util_sentence_group_vo_list = []
            for category, chars in sentence_groups.items():
                record_util_sentence_group_vo_list.append({
                    'GroupCategory': category,
                    'RecordUtilOracleCharVoList': chars
                })

            # 创建label.json条目
            if record_util_sentence_group_vo_list:
                label_entry = {
                    'Facsimile': facsimile_output_path,
                    'Rubbing': rubbing_output_path,
                    'RubbingName': output_name.split('.')[0],
                    'RecordUtilSentenceGroupVoList': record_util_sentence_group_vo_list
                }
                label_list.append(label_entry)

            total_split_count += 1

        file_count += 1

    # 保存 label.json
    print(f"\n保存 label.json 到 {output_label_json_path}...")
    with open(output_label_json_path, 'w') as f:
        json.dump(label_list, f, indent=2, ensure_ascii=False)

    print(f"完成!")
    print(f"处理了 {file_count} 个文件")
    print(f"总共 {total_inscription_boxes} 个 InscriptionSentence box")
    print(f"合并了 {total_merged_chars} 个非 InscriptionSentence 字符")
    print(f"总共切分了 {total_split_count} 个区域")
    print(f"生成了 {len(label_list)} 个label.json条目")

if __name__ == "__main__":
    main()
