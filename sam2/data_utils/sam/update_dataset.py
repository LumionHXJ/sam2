from sam2.data_utils.utils import filter_training_dataset

if __name__ == '__main__':
    with open("data/OBIMD_raw_hj/train.txt") as f:
        train_list = f.readlines()
        train_list = [line.strip() for line in train_list]
    filter_training_dataset(
        input_dir='data/OBIMD_sam/stage3/facsimile_json_full',
        output_dir='data/OBIMD_sam/stage3/facsimile_json',
        train_list=train_list,
        new_train_txt='data/OBIMD_sam/stage3/train.txt',
        iou_threshold=0.,
    )