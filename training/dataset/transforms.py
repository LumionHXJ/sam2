# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint


def hflip(datapoint, index):

    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: VideoDatapoint):
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint


class RandomMorphology:
    """
    Random morphology dilation and erosion for rubbing images.

    This transform applies random dilation or erosion to simulate variations in
    rubbing image quality. Dilation makes characters thicker, erosion makes them thinner.

    Args:
        p (float): Probability of applying transform. Default: 0.5
        kernel_size_range (tuple): Range of kernel sizes for morphology operations (min, max). Default: (3, 15)
        operation (str): Type of morphology operation - 'random', 'dilate', or 'erode'. Default: 'random'
        iterations_range (tuple): Range of iterations for morphology (min, max). Default: (1, 3)
        consistent_transform (bool): Whether to use same transform across frames. Default: True
    """
    def __init__(
        self,
        p=0.5,
        kernel_size_range=(3, 15),
        operation='random',
        iterations_range=(1, 3),
        consistent_transform=True
    ):
        self.p = p
        self.kernel_size_range = kernel_size_range
        self.operation = operation
        self.iterations_range = iterations_range
        self.consistent_transform = consistent_transform

        if self.operation not in ['random', 'dilate', 'erode']:
            raise ValueError(f"operation must be 'random', 'dilate', or 'erode', got {self.operation}")

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Generate random parameters once for all frames
            if random.random() < self.p:
                # Random operation
                if self.operation == 'random':
                    use_dilate = random.random() < 0.5
                elif self.operation == 'dilate':
                    use_dilate = True
                else:  # 'erode'
                    use_dilate = False

                # Random kernel size (must be odd)
                kernel_size = random.randint(*self.kernel_size_range)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                # Random iterations
                iterations = random.randint(*self.iterations_range)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op = cv2.MORPH_DILATE if use_dilate else cv2.MORPH_ERODE

                for img in datapoint.frames:
                    img.data = self._apply_morphology(img.data, op, kernel, iterations)
            return datapoint

        # Generate random parameters per frame
        for img in datapoint.frames:
            if random.random() < self.p:
                # Random operation
                if self.operation == 'random':
                    use_dilate = random.random() < 0.5
                elif self.operation == 'dilate':
                    use_dilate = True
                else:  # 'erode'
                    use_dilate = False

                # Random kernel size (must be odd)
                kernel_size = random.randint(*self.kernel_size_range)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                # Random iterations
                iterations = random.randint(*self.iterations_range)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op = cv2.MORPH_DILATE if use_dilate else cv2.MORPH_ERODE

                img.data = self._apply_morphology(img.data, op, kernel, iterations)

        return datapoint

    def _apply_morphology(self, img_data, op, kernel, iterations):
        """Apply morphology operation to image data."""
        if isinstance(img_data, PILImage.Image):
            # Convert PIL to numpy
            img_np = np.array(img_data)
            is_pil = True
        elif isinstance(img_data, torch.Tensor):
            # Convert tensor to numpy
            if img_data.dim() == 3:  # (C, H, W)
                img_np = img_data.permute(1, 2, 0).cpu().numpy()
            else:
                raise ValueError(f"Expected 3D tensor (C, H, W), got {img_data.dim()}D")
            is_pil = False
        else:
            raise TypeError(f"Unsupported type for morphology: {type(img_data)}")

        # Apply morphology to each channel (or use first channel for grayscale-like images)
        if img_np.shape[2] == 3:
            # RGB image - apply to each channel
            result = np.zeros_like(img_np)
            for c in range(3):
                result[:, :, c] = cv2.morphologyEx(img_np[:, :, c], op, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(img_np, op, kernel, iterations=iterations)

        # Convert back to original format
        if is_pil:
            return PILImage.fromarray(result.astype(np.uint8))
        else:
            return torch.from_numpy(result).permute(2, 0, 1).to(img_data.device)


class RandomRubbingDropout:
    """
    Random dropout that occludes foreground regions to simulate rubbing breaks and white spaces.

    This transform randomly adds white patches/drops to simulate natural breaks
    and white areas found in oracle bone rubbing images.

    Args:
        p (float): Probability of applying transform. Default: 0.3
        num_drops_range (tuple): Range of number of dropout patches to add (min, max). Default: (1, 5)
        drop_size_range (tuple): Range of dropout patch sizes (min, max). Default: (10, 50)
        consistent_transform (bool): Whether to use same transform across frames. Default: True
        drop_value (tuple): RGB value for dropout patches. Default: (255, 255, 255) (white)
    """
    def __init__(
        self,
        p=0.3,
        num_drops_range=(1, 5),
        drop_size_range=(10, 50),
        consistent_transform=True,
        drop_value=(255, 255, 255)
    ):
        self.p = p
        self.num_drops_range = num_drops_range
        self.drop_size_range = drop_size_range
        self.consistent_transform = consistent_transform
        self.drop_value = drop_value

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Generate random parameters once for all frames
            if random.random() < self.p:
                num_drops = random.randint(*self.num_drops_range)
                drop_params = []

                for _ in range(num_drops):
                    # Random drop size
                    drop_h = random.randint(*self.drop_size_range)
                    drop_w = random.randint(*self.drop_size_range)
                    # Random position will be determined per frame based on image size
                    drop_params.append((drop_h, drop_w))

                for img in datapoint.frames:
                    img.data = self._apply_dropout(img.data, drop_params)
            return datapoint

        # Generate random parameters per frame
        for img in datapoint.frames:
            if random.random() < self.p:
                num_drops = random.randint(*self.num_drops_range)
                drop_params = []

                for _ in range(num_drops):
                    # Random drop size
                    drop_h = random.randint(*self.drop_size_range)
                    drop_w = random.randint(*self.drop_size_range)
                    drop_params.append((drop_h, drop_w))

                img.data = self._apply_dropout(img.data, drop_params)

        return datapoint

    def _apply_dropout(self, img_data, drop_params):
        """Apply dropout patches to image data."""
        if isinstance(img_data, PILImage.Image):
            # Convert PIL to numpy
            img_np = np.array(img_data)
            is_pil = True
        elif isinstance(img_data, torch.Tensor):
            # Convert tensor to numpy
            if img_data.dim() == 3:  # (C, H, W)
                img_np = img_data.permute(1, 2, 0).cpu().numpy()
            else:
                raise ValueError(f"Expected 3D tensor (C, H, W), got {img_data.dim()}D")
            is_pil = False
        else:
            raise TypeError(f"Unsupported type for dropout: {type(img_data)}")

        h, w = img_np.shape[:2]

        for drop_h, drop_w in drop_params:
            # Random position
            y = random.randint(0, max(0, h - drop_h))
            x = random.randint(0, max(0, w - drop_w))

            # Apply dropout
            if len(img_np.shape) == 3:  # RGB
                img_np[y:y+drop_h, x:x+drop_w] = self.drop_value
            else:  # Grayscale
                img_np[y:y+drop_h, x:x+drop_w] = self.drop_value[0]

        # Convert back to original format
        if is_pil:
            return PILImage.fromarray(img_np.astype(np.uint8))
        else:
            return torch.from_numpy(img_np).permute(2, 0, 1).to(img_data.device)
