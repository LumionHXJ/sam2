# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        # Prototype encoder configuration for category-guided training
        prototype_root=None,
        prob_use_prototype_for_train=0.0,
        prob_use_prototype_for_eval=0.0,
        prob_use_prototype_as_only_for_train=0.0,
        prob_use_prototype_as_only_for_eval=0.0,
        prob_negative_sample_for_train=0.0,
        prob_negative_sample_for_eval=0.0,
        prob_negative_prototype_for_train=0.0,
        prob_negative_prototype_for_eval=0.0,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        # Prototype encoder configuration
        self.prototype_root = prototype_root
        self.prob_use_prototype_for_train = prob_use_prototype_for_train
        self.prob_use_prototype_for_eval = prob_use_prototype_for_eval
        self.prob_use_prototype_as_only_for_train = prob_use_prototype_as_only_for_train
        self.prob_use_prototype_as_only_for_eval = prob_use_prototype_as_only_for_eval
        self.prob_negative_sample_for_train = prob_negative_sample_for_train
        self.prob_negative_sample_for_eval = prob_negative_sample_for_eval
        self.prob_negative_prototype_for_train = prob_negative_prototype_for_train
        self.prob_negative_prototype_for_eval = prob_negative_prototype_for_eval

        # Initialize prototype loader if prototype_root is provided
        self.prototype_loader = None
        if prototype_root is not None:
            from training.dataset.prototype_loader import PrototypeLoader
            self.prototype_loader = PrototypeLoader(
                prototype_root=prototype_root,
                img_size=224,
                missing_as_zero=True,
            )

        # Build prototype pool for negative sampling
        self.prototype_pool = None
        if prototype_root is not None:
            self.prototype_pool = self._build_prototype_pool(prototype_root)
            logging.info(f"Built prototype pool with {len(self.prototype_pool)} UIDs")

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )


        # NEW: Prototype and negative sampling for category-guided training
        if self.training:
            prob_use_prototype = self.prob_use_prototype_for_train
            prob_negative_sample = self.prob_negative_sample_for_train
            prob_negative_prototype = self.prob_negative_prototype_for_train
        else:
            prob_use_prototype = self.prob_use_prototype_for_eval
            prob_negative_sample = self.prob_negative_sample_for_eval
            prob_negative_prototype = self.prob_negative_prototype_for_eval

        # Independent sampling for prototype and negative
        use_prototype = (prob_use_prototype > 0 and
                       self.rng.random() < prob_use_prototype)
        use_negative = (prob_negative_sample > 0 and
                      self.rng.random() < prob_negative_sample)
        use_negative_prototype = (prob_negative_prototype > 0 and
                                 self.rng.random() < prob_negative_prototype)

        # Check if prototypes are available (non-None labels)
        # If all labels are None, force fallback to classic box/point prompts
        has_valid_labels = False
        if use_prototype and hasattr(input, "labels") and input.labels is not None:
            # Check if any frame has valid (non-None) labels
            for frame_labels in input.labels:
                if frame_labels:  # if frame has any labels
                    if any(label is not None for label in frame_labels):
                        has_valid_labels = True
                        break

        # If no valid labels available, force fallback to classic prompts
        if use_prototype and not has_valid_labels:
            use_prototype = False

        # Store sampling decisions
        backbone_out["use_prototype"] = use_prototype
        backbone_out["use_negative"] = use_negative
        backbone_out["use_negative_prototype"] = use_negative_prototype

        # Extract labels from input for prototype loading
        if hasattr(input, "labels") and input.labels is not None:
            backbone_out["labels_per_frame"] = input.labels
        else:
            backbone_out["labels_per_frame"] = None

        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        # NEW: Handle prototype inputs for category-guided training
        backbone_out["prototype_embeddings_per_frame"] = {}  # {frame_idx: [B, 1, C]}
        # Mark frames that use negative prototypes (to set gt_mask to all zeros)
        backbone_out["negative_prototype_frames"] = set()

        for t in init_cond_frames:
            # Handle prototype inputs
            if use_prototype and (backbone_out.get("labels_per_frame") is not None):
                # Get labels for this frame
                frame_labels = backbone_out["labels_per_frame"][t] if t < len(backbone_out["labels_per_frame"]) else []

                if use_negative_prototype:
                    # Use negative prototypes instead of positive ones
                    # Determine number of negatives (1:1 or fewer)
                    valid_labels = [l for l in frame_labels if l is not None]
                    num_negatives = min(len(valid_labels), self.rng.integers(0, len(valid_labels) + 1, endpoint=True))
                    if num_negatives > 0:
                        # Sample negative prototypes
                        negative_labels = self._sample_negative_prototypes(frame_labels, num_negatives)
                        # Encode negative prototypes
                        prototype_embeddings = self._prepare_prototype_batch(
                            negative_labels, device=torch.device("cuda")
                        )
                        if prototype_embeddings is not None:
                            backbone_out["prototype_embeddings_per_frame"][t] = prototype_embeddings
                            # Mark this frame as using negative prototypes
                            backbone_out["negative_prototype_frames"].add(t)
                            # Set gt_mask to all zeros (background) for negative prototypes
                            B, _, H, W = gt_masks_per_frame[t].shape
                            gt_masks_per_frame[t] = torch.zeros(B, 1, H, W, device=gt_masks_per_frame[t].device, dtype=torch.bool)
                else:
                    # Use positive prototypes (existing logic)
                    prototype_embeddings = self._prepare_prototype_batch(
                        frame_labels, device=torch.device("cuda")
                    )
                    if prototype_embeddings is not None:
                        backbone_out["prototype_embeddings_per_frame"][t] = prototype_embeddings

            # Handle negative sampling
            if use_negative and (backbone_out.get("labels_per_frame") is not None):
                # Sample negative prompts from background
                negative_prompts = self._sample_negative_prompts(gt_masks_per_frame[t])
                # Check if negative prompts were successfully generated
                if negative_prompts["point_coords"].numel() > 0:
                    # Set gt_mask to all zeros (background) for negative prompts
                    B, _, H, W = gt_masks_per_frame[t].shape
                    gt_masks_per_frame[t] = torch.zeros(B, 1, H, W, device=gt_masks_per_frame[t].device, dtype=torch.bool)
                # Merge with existing point inputs (if any)
                existing_point_inputs = backbone_out["point_inputs_per_frame"].get(t, None)
                if existing_point_inputs is not None:
                    # Concatenate existing points with negative points
                    merged_points = concat_points(
                        existing_point_inputs,
                        negative_prompts["point_coords"],
                        negative_prompts["point_labels"]
                    )
                    backbone_out["point_inputs_per_frame"][t] = merged_points
                else:
                    backbone_out["point_inputs_per_frame"][t] = negative_prompts

            if not use_pt_input: # for misalign correction
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
                # sample box input for second frame
                points, labels = sample_box_points(
                    gt_masks_per_frame[1],
                 )
                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][1] = point_inputs
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def _prepare_prototype_batch(self, labels, device):
        """
        Load and encode prototype character images into spatial tokens.

        Args:
            labels: List of label strings for batch.
            device: Torch device to place tensors on.

        Returns:
            torch.Tensor: Spatial token embeddings with shape [B, num_patches, C],
                         where num_patches=196 for 14x14 spatial grid, or None if no prototype loader.
        """
        if self.prototype_loader is None:
            return None

        # Load prototype images
        prototype_imgs = self.prototype_loader.load_prototypes(labels)
        prototype_imgs = prototype_imgs.to(device)

        # Encode prototypes into spatial tokens
        with torch.no_grad():
            spatial_embeddings = self.prototype_encoder(prototype_imgs)  # [B, 196, 256]

        return spatial_embeddings  # [B, 196, 256]

    def _sample_negative_prompts(self, gt_masks):
        """
        Sample negative prompts (simulating incorrect clicks/boxes).

        Robustness improvement: Reject generation if no foreground exists.
        Uses either point or box based on prob_to_use_box_input_for_train.

        Args:
            gt_masks: Ground truth masks [B, 1, H, W].

        Returns:
            dict: Negative prompts with "point_coords" and "point_labels".
                  Returns empty tensors if no foreground found.
        """
        from sam2.utils.misc import mask_to_box
        B, _, H, W = gt_masks.shape

        # Sample a fixed number of prompts for all batches
        num_prompts = 1  # Only sample one negative point

        points_list = []
        labels_list = []

        # Use box or point based on prob_to_use_box_input_for_train
        use_box = self.rng.random() < self.prob_to_use_box_input_for_train

        for b in range(B):
            mask = gt_masks[b, 0].cpu()

            # Check if there is any foreground
            fg_indices = torch.where(mask)
            if fg_indices[0].numel() == 0:
                # No foreground: reject generation, return empty for this batch
                continue

            if use_box:
                # Sample negative box: random box anywhere (not based on gt mask)
                # Label 2=top_left, 3=bottom_right for box corners
                x1, x2 = self.rng.integers(0, W, 2, endpoint=False)
                y1, y2 = self.rng.integers(0, H, 2, endpoint=False)
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                points = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32)
                labels = torch.tensor([[2, 3]], dtype=torch.long)  # box corner labels
            else:
                # Sample negative points: random points anywhere
                # Simulating incorrect clicks with label=1 (foreground) but gt_mask=all-0
                y_coords = self.rng.integers(0, H, num_prompts)
                x_coords = self.rng.integers(0, W, num_prompts)

                points = torch.tensor([
                    [x_coords[i], y_coords[i]]
                    for i in range(num_prompts)
                ], dtype=torch.float32).unsqueeze(0)

                labels = torch.ones(1, num_prompts, dtype=torch.long)  # label=1 for point

            points_list.append(points)
            labels_list.append(labels)

        if not points_list:
            # No valid batches, return empty tensors
            return {
                "point_coords": torch.zeros(B, 0, 2, device=gt_masks.device),
                "point_labels": torch.zeros(B, 0, dtype=torch.long, device=gt_masks.device)
            }

        # Pad to maintain batch size consistency
        while len(points_list) < B:
            points_list.append(torch.zeros(1, 0, 2))
            labels_list.append(torch.zeros(1, 0, dtype=torch.long))

        return {
            "point_coords": torch.cat(points_list, dim=0).to(gt_masks.device),
            "point_labels": torch.cat(labels_list, dim=0).to(gt_masks.device)
        }

    def _build_prototype_pool(self, prototype_root: str):
        """
        Build a list of all available prototype UIDs from the directory.

        Args:
            prototype_root: Path to the prototype root directory.

        Returns:
            List[str]: List of prototype UIDs.
        """
        import os
        pool = []
        for filename in os.listdir(prototype_root):
            # Extract UID from filename (remove extension)
            uid, _ = os.path.splitext(filename)
            if uid:  # Skip empty names
                pool.append(uid)
        return pool

    def _sample_negative_prototypes(
        self,
        positive_labels,
        num_negatives
    ):
        """
        Sample negative prototype UIDs not in the current frame.

        Args:
            positive_labels: List of positive label strings for the current frame.
            num_negatives: Number of negative prototypes to sample.

        Returns:
            List[Optional[str]]: List of negative prototype UIDs.
        """
        if self.prototype_pool is None:
            return [None] * num_negatives

        # Get valid positive labels (non-None)
        valid_positives = [label for label in positive_labels if label is not None]
        if not valid_positives:
            return [None] * num_negatives

        # Create a set of labels to exclude
        exclude_set = set(valid_positives)

        # Filter pool to exclude current frame labels
        available_negatives = [uid for uid in self.prototype_pool if uid not in exclude_set]

        if not available_negatives:
            return [None] * num_negatives

        # Sample num_negatives from available pool (with replacement if needed)
        selected = self.rng.choice(
            len(available_negatives),
            min(num_negatives, len(available_negatives)),
            replace=len(available_negatives) < num_negatives
        )

        return [available_negatives[i] for i in selected]

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs
