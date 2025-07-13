from tqdm import tqdm
import os
import numpy as np
import json
ious = []
os.makedirs("data/OBIMD_HJ_diff/fascimile_json_round2_train", exist_ok=True)
for file in tqdm(os.listdir("data/OBIMD_HJ_diff/fascimile_json_round1_out")):
    with open(os.path.join("data/OBIMD_HJ_diff/fascimile_json_round1_out", file)) as f:
        data = json.load(f)
    data['annotations'] = [d for d in data['annotations'] if d['predicted_iou'] >= 0.7]
    if len(data['annotations']) == 0:
        continue
    with open(os.path.join("data/OBIMD_HJ_diff/fascimile_json_round2_train", file), 'w') as f:
        json.dump(data, f, indent=2)
with open('data/OBIMD_HJ_diff/train_round2.txt', mode='w') as f:
    for file in sorted(os.listdir("data/OBIMD_HJ_diff/fascimile_json_round2_train")):
        f.write(file.replace('.json', '') + '\n')