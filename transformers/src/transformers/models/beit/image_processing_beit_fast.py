# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for Beit."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    is_torch_tensor,
    make_list_of_images,
    pil_torch_interpolation_mapping,
    validate_kwargs,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring


class BeitFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    do_reduce_labels: Optional[bool]


@auto_docstring
class BeitImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = True
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False
    valid_kwargs = BeitFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[BeitFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to save support of deprecated `reduce_labels` in old configs
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in image_processor_dict:
            image_processor_dict["do_reduce_labels"] = image_processor_dict.pop(
                "reduce_labels"
            )
        return super().from_dict(image_processor_dict, **kwargs)

    def reduce_label(self, labels: list["torch.Tensor"]):
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
            label = label - 1
            label = torch.where(
                label == 254, torch.tensor(255, dtype=label.dtype), label
            )
            labels[idx] = label

        return label

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        if do_reduce_labels:
            images = self.reduce_label(images)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_images = (
            torch.stack(processed_images, dim=0) if return_tensors else processed_images
        )
        return processed_images

    def _preprocess_images(
        self,
        images,
        **kwargs,
    ):
        """Preprocesses images."""
        kwargs["do_reduce_labels"] = False
        processed_images = self._preprocess(images=images, **kwargs)
        return processed_images

    def _preprocess_segmentation_maps(
        self,
        segmentation_maps,
        **kwargs,
    ):
        """Preprocesses segmentation maps."""
        processed_segmentation_maps = []
        for segmentation_map in segmentation_maps:
            segmentation_map = self._process_image(
                segmentation_map,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            if segmentation_map.ndim == 2:
                segmentation_map = segmentation_map[None, ...]

            processed_segmentation_maps.append(segmentation_map)

        kwargs["do_normalize"] = False
        kwargs["do_rescale"] = False
        kwargs["input_data_format"] = ChannelDimension.FIRST
        processed_segmentation_maps = self._preprocess(
            images=processed_segmentation_maps, **kwargs
        )

        processed_segmentation_maps = processed_segmentation_maps.squeeze(1)

        processed_segmentation_maps = processed_segmentation_maps.to(torch.int64)
        return processed_segmentation_maps

    def __call__(self, images, segmentation_maps=None, **kwargs):
        # Overrides the `__call__` method of the `Preprocessor` class such that the images and segmentation maps can both
        # be passed in as positional arguments.
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs: Unpack[BeitFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self.valid_kwargs.__annotations__.keys(),
        )
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        # Prepare segmentation maps
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(
                images=segmentation_maps, expected_ndims=2
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample]
            if isinstance(resample, (PILImageResampling, int))
            else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        images = self._preprocess_images(
            images=images,
            **kwargs,
        )
        data = {"pixel_values": images}

        if segmentation_maps is not None:
            segmentation_maps = self._preprocess_segmentation_maps(
                segmentation_maps=segmentation_maps,
                **kwargs,
            )
            data["labels"] = segmentation_maps

        return BatchFeature(data=data)

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple]] = None
    ):
        """
        Converts the output of [`BeitForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`BeitForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # TODO: add support for other frameworks
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [
                semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
            ]

        return semantic_segmentation


__all__ = ["BeitImageProcessorFast"]
