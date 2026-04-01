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

#### Raw Annotation Format (`data/OBIMD_raw_hj/label_filtered.json`)

JSON array with each element representing one image:

```json
{
  "Facsimile": "/path/to/facsimile_image.jpg",
  "Rubbing": "/path/to/rubbing_image.jpg",
  "RubbingName": "H2",
  "RecordUtilSentenceGroupVoList": [
    {
      "GroupCategory": "InscriptionSentence1",
      "RecordUtilOracleCharVoList": [
        {
          "Position": "x,y,w,h",  // Bounding box in x,y,width,height format
          "OrderNumber": 5,
          "SeatFont": 0,
          "Mark": -1,
          "Label": "character_label",
          "SubLabel": "sub_character_label"
        },
        // ... more characters
      ]
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

- `sam2/data_utils/stage1/sam_on_gt_bbox.py`: Uses SAM to generate masks from GT bounding boxes, with multi-GPU support (4 GPUs by default)
- `sam2/data_utils/stage1/update_dataset.py`: Merge and update dataset between stages
- `sam2/data_utils/filter_raw_data/to_sa1b_format.py`: Convert raw data to SA-1B format
- `sam2/data_utils/convert_sa1b_to_yolo.py`: Convert SA-1B format to YOLO format

### Data Utils Functions

Key functions in `sam2/data_utils/utils.py`:
- `get_tight_box(mask)`: Get tight bounding box from mask (returns x,y,w,h)
- `calculate_stability_score(masks)`: Calculate SAM stability score
- `calc_box_iou(box1, box2)`: Calculate IoU between two bounding boxes
- `calc_mask_coverage(mask1, mask2)`: Calculate pixel coverage of mask1 by mask2

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
