# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from sam2.modeling.backbones.prototype_encoder import PrototypeEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms


class PrototypeLoader:
    """Loads prototype character images from the database.

    This loader retrieves prototype character images (字头) from the database
    and preprocesses them for use as additional prompts in SAM 2.
    """

    def __init__(
        self,
        prototype_root: str,
        img_size: int = 224,
        missing_as_zero: bool = True,
    ):
        """
        Args:
            prototype_root: Path to directory containing prototype character images.
            img_size: Size to resize prototype images to (default: 224).
            missing_as_zero: If True, return zero tensor for missing prototypes.
        """
        self.prototype_root = prototype_root
        self.img_size = img_size
        self.missing_as_zero = missing_as_zero

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_single_prototype(self, label: Optional[str]) -> Optional[torch.Tensor]:
        """Load a single prototype image.

        Returns:
            torch.Tensor if label is valid, None if label is None.
        """
        if label is None:
            return None

        # Try both .png and .jpg extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(self.prototype_root, f"{label}{ext}")
            if os.path.exists(img_path):
                try:
                    img = PILImage.open(img_path).convert('RGB')
                    return self.transform(img)
                except Exception as e:
                    logging.warning(f"Warning: Failed to load {img_path}: {e}")
                    break

        # If not found, return None
        logging.warning(f"Warning: Prototype image not found for label '{label}'")
        return None

    def load_prototypes(self, labels: List[Optional[str]]) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Load prototype images for given labels.

        Args:
            labels: List of label strings (or None for missing labels).

        Returns:
            tuple: (prototype_images, valid_indices)
                - prototype_images: Batch of prototype images with shape [B, 3, H, W],
                  or None if no valid prototypes found.
                - valid_indices: List of indices in the original list that correspond
                  to valid (non-None) labels.
        """
        batch = []
        valid_indices = []
        for i, label in enumerate(labels):
            prototype = self._load_single_prototype(label)
            if prototype is not None:
                batch.append(prototype)
                valid_indices.append(i)

        if not batch:
            return None, []

        return torch.stack(batch), valid_indices


class SAM2ImagePredictorWithPrototype:
    """SAM2ImagePredictor with prototype character support.

    This predictor extends SAM2ImagePredictor to support prototype character
    images (字头) as additional prompts. Prototype images are loaded from the
    database and encoded using a Vision Transformer backbone.
    """

    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        prototype_root: str = "data/镜原数据库/字头图像",
        prototype_encoder_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            sam_model: The SAM2 model to use for mask prediction.
            mask_threshold: The threshold to use when converting mask logits
                to binary masks. Masks are thresholded at 0 by default.
            max_hole_area: If max_hole_area > 0, we fill small holes in up to
                the maximum area of max_hole_area in low_res_masks.
            max_sprinkle_area: If max_sprinkle_area > 0, we remove small sprinkles up to
                the maximum area of max_sprinkle_area in low_res_masks.
            prototype_root: Path to directory containing prototype character images.
            prototype_encoder_config: Configuration for PrototypeEncoder. If None,
                uses default config.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        # Prototype support
        self.prototype_root = prototype_root
        self.prototype_loader = PrototypeLoader(prototype_root)

        # Initialize prototype encoder
        if prototype_encoder_config is None:
            prototype_encoder_config = {
                "embed_dim": 192,
                "output_dim": 256,
                "pretrained": True,
                "freeze_backbone": False,
                "model_name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            }

        # Check if model has prototype_encoder attribute
        if hasattr(self.model, 'prototype_encoder'):
            self.prototype_encoder = self.model.prototype_encoder
        else:
            # Create a new prototype encoder
            self.prototype_encoder = PrototypeEncoder(**prototype_encoder_config)
            logging.warning("Model does not have prototype_encoder attribute, using new instance")

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictorWithPrototype":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictorWithPrototype): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, PILImage.Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format.
            The image should be in HWC format if np.ndarray, or WHC format if PIL Image
            with pixel values in [0, 255].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, PILImage.Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format.
            The image should be in HWC format if np.ndarray with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        prototype_uids: Optional[List[Optional[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized
            to the range [0,1] and point_coords is expected to be wrt. image dimensions.
          prototype_uids (list of str or None): A list of prototype character UIDs.
            Can be used independently or together with box prompts. None values in the
            list are filtered out.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Note: prototype_uids can now be used independently
        # No box requirement validation needed

        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        # Load and encode prototypes if provided
        prototype_embeddings = None
        if prototype_uids is not None:
            prototype_embeddings = self._load_and_encode_prototypes(prototype_uids)

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            prototype_embeddings=prototype_embeddings,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def batch_predict(
        self,
        batchsize: int = 16,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        prototype_uids_batch: Optional[List[List[Optional[str]]]] = None,
        *args, **kwargs
    ):
        """Batch prediction with prototype support.

        Arguments:
          batchsize: Number of samples per batch.
          point_coords: Point coordinates for all samples.
          point_labels: Point labels for all samples.
          box: Bounding boxes for all samples.
          mask_input: Mask inputs for all samples.
          prototype_uids_batch: List of prototype UID lists, one per sample.
          *args, **kwargs: Additional arguments passed to predict().
        """
        all_masks = []
        all_scores = []
        all_logits = []

        N = len(box) if box is not None else len(point_coords)

        # Validate prototype_uids_batch consistency
        if prototype_uids_batch is not None:
            if len(prototype_uids_batch) != N:
                raise ValueError(
                    f"prototype_uids_batch length ({len(prototype_uids_batch)}) must match "
                    f"number of samples ({N})"
                )

        # Process in batches
        for i in range(0, N, batchsize):
            # Get current batch
            batch_boxes = box[i:i+batchsize] if box is not None else None
            batch_point_coords = point_coords[i:i+batchsize] if point_coords is not None else None
            batch_point_labels = point_labels[i:i+batchsize] if point_labels is not None else None
            batch_mask_input = mask_input[i:i+batchsize] if mask_input is not None else None

            # Get prototype UIDs for this batch
            batch_prototype_uids = None
            if prototype_uids_batch is not None:
                batch_prototype_uids = prototype_uids_batch[i:i+batchsize]
                # Flatten for single batch prediction
                if len(batch_prototype_uids) == 1:
                    batch_prototype_uids = batch_prototype_uids[0]

            masks, scores, logits = self.predict(
                point_coords=batch_point_coords,
                point_labels=batch_point_labels,
                box=batch_boxes,
                mask_input=batch_mask_input,
                prototype_uids=batch_prototype_uids,
                *args, **kwargs
            )

            if len(masks) == 1:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]

            all_masks.append(masks)
            all_scores.append(scores)
            all_logits.append(logits)

        all_masks = np.concatenate(all_masks, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)

        if len(all_masks) == 1:
            all_masks = all_masks[0]
            all_scores = all_scores[0]
            all_logits = all_logits[0]

        return all_masks, all_scores, all_logits

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        prototype_uids_batch: Optional[List[List[Optional[str]]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode,
        when the model is expected to generate predictions on multiple images.

        Returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []

        # Validate prototype_uids_batch consistency
        if prototype_uids_batch is not None:
            if len(prototype_uids_batch) != num_images:
                raise ValueError(
                    f"prototype_uids_batch length ({len(prototype_uids_batch)}) must match "
                    f"number of images ({num_images})"
                )

        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            prototype_uids = (
                prototype_uids_batch[img_idx] if prototype_uids_batch is not None else None
            )

            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )

            # Load and encode prototypes if provided
            prototype_embeddings = None
            if prototype_uids is not None:
                prototype_embeddings = self._load_and_encode_prototypes(prototype_uids)

            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
                prototype_embeddings=prototype_embeddings,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
        prototype_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          img_idx (int): Index of the image in batch mode.
          prototype_embeddings (torch.Tensor or None): Prototype character embeddings
            with shape [B, N, embed_dim] where N is number of prototypes.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
            prototype_embeddings=prototype_embeddings,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def _load_and_encode_prototypes(
        self,
        uids: List[Optional[str]],
    ) -> Optional[torch.Tensor]:
        """Load and encode prototype images.

        Args:
            uids: List of prototype character UIDs (may contain None).

        Returns:
            torch.Tensor: Prototype embeddings with shape [1, N, embed_dim],
                where N is the total number of input UIDs (including None values).
                None positions are filled with learnable mask embeddings.
                Returns None if no valid prototypes found.
        """
        # Load prototype images (filters None labels)
        prototype_images, valid_indices = self.prototype_loader.load_prototypes(uids)

        if prototype_images is None:
            return None

        prototype_images = prototype_images.to(self.device)

        # Encode prototypes
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            valid_embeddings = self.prototype_encoder(prototype_images)  # [N_valid, embed_dim]

        # Initialize full batch with learnable mask embeddings for None positions
        num_total = len(uids)
        embed_dim = valid_embeddings.shape[1]
        device = valid_embeddings.device

        # Use learnable mask embeddings from prompt encoder for None positions
        mask_embedding = self.model.sam_prompt_encoder.no_mask_embed.weight.squeeze(0)  # [embed_dim]

        # Create full batch embeddings with [1, N_total, embed_dim]
        # Initialize with mask_embedding for all positions
        full_embeddings = mask_embedding.unsqueeze(0).unsqueeze(0).repeat(1, num_total, 1)
        # Fill valid positions with encoded prototypes
        for i, valid_idx in enumerate(valid_indices):
            full_embeddings[0, valid_idx] = valid_embeddings[i]

        return full_embeddings

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
