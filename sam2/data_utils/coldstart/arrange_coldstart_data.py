from sam2.data_utils.utils import update_training_dataset

if __name__ == '__main__':
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = f.readlines()
        train_list = [line.strip() for line in train_list]
    update_training_dataset(
        input_dir='data/OBIMD_raw_hj/facsimile_json_coldstart',
        output_dir='data/OBIMD_iou0.6/coldstart/facsimile_json',
        accept_dir=None,
        train_list=train_list,
        new_train_txt='data/OBIMD_iou0.6/coldstart/train.txt',
        iou_threshold=0.6,
    )
