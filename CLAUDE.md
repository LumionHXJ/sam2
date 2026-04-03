# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
pip install -e ".[notebooks]"
# For training, also install dev dependencies:
pip install -e ".[dev]"
```

### CUDA Extension

- Builds by default, but installation continues on failure (post-processing may be skipped)
- Skip CUDA build: `SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"`
- Force CUDA build: `SAM2_BUILD_ALLOW_ERRORS=0 pip install -e ".[notebooks]"`
- Rebuild if needed: `python setup.py build_ext --inplace`

## Model Architecture

SAM 2 uses a transformer-based architecture with streaming memory for video processing:
- **Image Encoder**: Hiera backbone (hierarchical vision transformer) with FPN neck
- **Memory Encoder**: Encodes mask and image features into memory tokens
- **Memory Attention**: Cross-attends current frame features to memory tokens
- **SAM Mask Decoder**: Predicts masks from image features and prompts
- Model configs are in `sam2/configs/sam2.1/` and `sam2/configs/sam2/`

Key config options:
- `compile_image_encoder`: Enable torch.compile for speedup
- `vos_optimized`: Use SAM2VideoPredictorVOS for faster video inference
- `use_high_res_features_in_sam`: Use high-res features in mask decoder

## SAM 2 Inference

### Image Prediction

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

predictor = SAM2ImagePredictor(
    build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt")
)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords, point_labels, box=input_box)
```

### Video Prediction

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
    vos_optimized=True  # for faster VOS
)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video)
    frame frame_idx, obj_ids, masks = predictor.add_new_points_or_box(state, prompts)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # Process masks frame by frame
```

## Training

**Important**: The `-c` config path is relative to `configs/` directory (not the project root), because Hydra's config search path is set to `pkg://sam2` which resolves to the `configs/` directory. So use `configs/sam2.1_training/...` instead of `sam2/configs/sam2.1_training/...`.

### Single GPU Training

```bash
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

### Multi-Node SLURM Training

```bash
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 1 \
    --num-gpus 8 \
    --num-nodes 2 \
    --partition $PARTITION
```

### Training Configs

- `configs/sam2.1_training_stage1/`: Stage 1 training configs
- `configs/sam2.1_training_stage2/`: Stage 2 training configs (pretrain, sft)
- `configs/sam2.1_baseline/`: Baseline configs for OBIMD dataset (raw, charformer, segformer)

Training logs and checkpoints are saved to `sam2_logs/` by default (configurable via `experiment_log_dir`).

## OBIMD (Oracle Bone Character) Data Pipeline

This repo contains a custom data pipeline for Oracle Bone Character mask segmentation.

### Data Format

#### Raw Annotation Format (`data/OBIMD_raw_hj/label.json` / `label_filtered.json`)

JSON array with each element representing one image. This is the raw annotation format for the Oracle Bone Character dataset.

```json
{
  "Facsimile": "/path/to/facsimile_image.jpg",  // 摩本图像路径
  "Rubbing": "/path/to/rubbing_image.jpg",      // 拓片图像路径
  "RubbingName": "H2",                          // 拓片名称标识符
  "RecordUtilSentenceGroupVoList": [             // 句子组数组
    {
      "GroupCategory": "InscriptionSentence1",   // 句子组类别
      "RecordUtilOracleCharVoList": [             // 该组中的甲骨文字符注解
        {
          "Position": "x,y,w,h",      // 边界框 (x=左上角x, y=左上角y, w=宽度, h=高度)
          "OrderNumber": 5,            // 字符在句子中的显示顺序
          "SeatFont": 0,               // 字体座位信息
          "Mark": -1,                  // 标记标志 (-1=未标记)
          "Label": "character_label",   // 字符标签 (未标注时为None). 对应字头 uid，可在 data/镜原数据库/字头图像 找到原型字
          "SubLabel": "sub_character_label"  // 子标签 (通常与Label相同)
        },
        // ... more characters
      ]
    }
  ]
}
```

**Label File Variants**:
- `label.json`: 完整原始注解 (~33.9MB)
- `label_filtered.json`: 过滤后注解 (移除空边界框, ~30.4MB)
- `label_filt_train.json`: 训练用过滤版本 (label_filtered.json的子集)

**Usage Example**:
```python
from sam2.data_utils.utils import load_raw_annotations

# Load raw annotations from label.json
data_lookup = load_raw_annotations(
    "data/OBIMD_raw_hj/label_filtered.json",
    "data/OBIMD_raw_hj/facsimile",
    ignore_null=True  # 忽略标签为None的字符
)

# Access annotations for a specific image
image_name = "h00002"  # 不包含.jpg扩展名
oracle_chars = data_lookup[image_name]

# Process each character
for char in oracle_chars:
    x, y, w, h = list(map(int, char['Position'].split(',')))
    label = char['Label']
    print(f"Character at ({x},{y}) with size {w}x{h}, label: {label}")
```

#### SA-1B Format

SA-1B format is the intermediate annotation format bridging raw annotations and training-ready formats. Each character is saved as a separate JSON file in the directory structure: `data/OBIMD_iou0.6/SA1B/{image_name}/{char_id}.json`.

```json
{
  "image_info": {
    "image_id": "h00002",
    "file_name": "/path/to/image.jpg",
    "height": 1024,
    "width": 1024
  },
  "annotations": [
    {
      "id": 0,
      "area": 12345,
      "bbox": [x, y, w, h],  // 紧凑边界框 (从mask计算)
      "segmentation": {
        "size": [height, width],
        "counts": "RLE_encoded_string"  // COCO RLE格式
      },
      "crop_box": [x, y, w, h],  // 原始裁剪框
      "Label": "character_label",  // 字符标签
      "SubLabel": "sub_character_label",  // 子标签
      "Mark": -1  // 标记标志
    }
  ]
}
```

#### Processed Mask Format (`data/OBIMD_iou0.6/stageN/facsimile_json/*.json`)

COCO-style JSON format with SAM predictions:

```json
{
  "image_info": {
    "image_id": "h33215",
    "width": 496,
    "height": 832,
    "file_name": "h33215.jpg"
  },
  "annotations": [
    {
      "id": 314,
      "bbox": [x, y, w, h],  // Tight bounding box from mask
      "area": 585,
      "segmentation": {
        "size": [height, width],
        "counts": "RLE_encoded_string"  // COCO RLE format
      },
      "predicted_iou": 0.5657,  // IOU between rubbing and facsimile masks
      "stability_score": 0.8018,  // SAM stability score
      "crop_box": [x, y, w, h]  // Original crop box
    },
    // ... more annotations
  ]
}
```

### Directory Structure

**Raw Data**:
- `data/OBIMD_raw_hj/rubbing`: Rubbing images (X-ray images)
- `data/OB`IMD_raw_hj/facsimile`: Facsimile images (rubbing images)
- `data/OBIMD_raw_hj/label_filtered.json`: Raw annotations
- `data/OBIMD_raw_hj/train.txt`: List of training image IDs

**Processed Data** (stage-wise):
- `data/OBIMD_iou0.6/stageN/facsimile_json`: Processed COCO-format masks
- `data/OBIMD_iou0.6/stageN/train.txt`: Training image list for stage N

**Stage Pipeline**:
- Stage 1: SAM prediction on ground truth bounding boxes
- Stage 2: Filtered predictions with IOU threshold
- Stage 3: Final filtered dataset

### Key Scripts

#### Root Directory Scripts (`sam2/data_utils/`)

- `yolo_predict.py`: Run YOLO detection on rubbing images to generate bounding box predictions
- `sam_on_yo_prediction.py`: Generate SAM masks from YOLO box predictions
- `character_structural_sim.py`: Calculate PSNR and SSIM metrics for character structural similarity
- `generate_cls_on_gt_bbox.py`: Generate classification dataset using ground truth bounding boxes
- `sam_on_gt_bbox_hd.py`: Segmentation for HD (high definition) dataset
- `add_label_to_sa1b.py`: Add character labels to existing SA-1B format files
- `convert_sa1b_to_yolo.py`: Convert SA-1B format to YOLO format for object detection training
- `crop_hd_to_test.py`: Crop HD data for testing purposes

#### Filter Raw Data (`sam2/data_utils/filter_raw_data/`)

- `to_sa1b_format.py`: Convert raw label.json annotations to SA-1B format by extracting facsimile masks using character bounding boxes. Handles overlapping character masks using connected component analysis.
- `filter_empty_box.py`: Clean raw annotations by filtering out empty bounding boxes. Verifies if character masks exist in facsimile images and handles borderline cases with character position shifting.

#### Coldstart Data (`sam2/data_utils/coldstart/`)

- `arrange_coldstart_data.py`: Prepare cold-start dataset by organizing data for few-shot or zero-shot scenarios
- `generate_align_data_by_shift.py`: Align masks using shift optimization to improve mask-to-image correspondence

#### Stage 1 Processing (`sam2/data_utils/stage1/`)

- `sam_on_gt_bbox.py`: Uses SAM model to generate masks from ground truth bounding boxes in label.json format. Processes images in parallel across GPUs (4 GPUs by default). Generates COCO-style annotations with predicted masks.
- `update_dataset.py`: Merge and update dataset between different stages of the data processing pipeline

#### SAM Processing (`sam2/data_utils/sam/`)

- `sam_self_iterate.py`: Self-iterative SAM refinement - iteratively improve mask predictions
- `update_dataset.py`: Filter SAM-generated dataset to remove low-quality predictions

### Data Utils Functions

Key functions in `sam2/data_utils/utils.py`:

#### Bounding Box Operations
- `get_tight_box(mask)`: Get tight bounding box from binary mask (returns x,y,w,h)
- `calc_box_iou(box1, box2)`: Calculate IoU between two bounding boxes
- `calc_box_isin(box1, box2)`: Check if box1 is inside box2
- `calc_box_area(box)`: Calculate area of a bounding box

#### Mask Operations
- `calculate_stability_score(masks)`: Calculate SAM stability score for masks
- `calc_mask_coverage(mask1, mask2)`: Calculate pixel coverage of mask1 by mask2
- `calc_mask_iou(mask1, mask2)`: Calculate IoU between two masks
- `combine_masks(mask_list)`: Combine multiple masks using logical OR

#### Data Loading
- `load_raw_annotations(data_path, segmap_root, ignore=False)`: Load raw annotations from label.json format, returns dictionary mapping image name to list of oracle characters
- `load_null_annotations(data_path)`: Load only characters with None labels from label.json

#### Dataset Processing
- `filter_training_dataset(dataset_path, threshold=0.6)`: Filter dataset based on predicted_iou threshold
- `update_training_dataset(old_dataset, new_dataset, overlap_mode)`: Merge and update training datasets between stages

#### Image Processing
- `rotate_image(image, angle)`: Rotate image by specified angle
- `extract_connected_component_opencv(mask)`: Extract connected components from binary mask using OpenCV

#### Evaluation
- `calculate_mask_ap(pred_masks, gt_masks, iou_thresholds)`: Calculate Average Precision for mask predictions

## YOLO Integration

YOLO is used for object detection on character masks:

```python
# training/train_yolo.py
from ultralytics import YOLO

model = YOLO("checkpoints/yolov12x.pt")
model.train(
    data="sam2/configs/yolo/OBIMD.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="0,1,2,3"
)
```

YOLO config: `sam2/configs/yolo/OBIMD.yaml` (defines dataset paths and classes)

## Data Transforms

Video/Image transforms for training are defined using Hydra target pattern:
- `RandomHorizontalFlip`: Horizontal flip
- `RandomAffine`: Random affine transformations
- `RandomResizeAPI`: Resize to specified resolution
- `ColorJitter` Color augmentation
- `ToTensorAPI`: Convert to tensor
- `NormalizeAPI`: Normalize with ImageNet stats

## Important Notes

1. **Do not run Python from the parent directory** of this repo (will cause import errors)
2. Use `torch.inference_mode()` and `torch.autocast("cuda", dtype=torch.bfloat16)` for better inference performance
3. Data links are symlinks: `data/` -> `/datab/huxingjian/data/oraclebone/OBICHARiot`
4. Logs link: `sam2_logs/` -> `/datab/huxingjian/experiments/obichariot`
5. This project is a customized version for Oracle Bone Character segmentation - base SAM 2 code is in `sam2/modeling/` and `sam2/` (predictors)
