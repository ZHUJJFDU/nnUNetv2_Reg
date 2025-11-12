from typing import Union, List, Tuple

import os
import json
import numpy as np
import torch
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy as nnUNetDataset


class RegnnUNetDataset(nnUNetDataset):
    def __init__(self,
                 folder: str,
                 regression_values_file: str,
                 regression_key: str = "bulla_thickness",
                 case_identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, case_identifiers, folder_with_segs_from_previous_stage)
        self.regression_values_file = regression_values_file
        self.regression_key = regression_key
        self.regression_values = self._load_regression_values()

    def _load_regression_values(self):
        if not isfile(self.regression_values_file):
            raise FileNotFoundError(f"Regression values file not found: {self.regression_values_file}")
        with open(self.regression_values_file, 'r', encoding='utf-8-sig') as f:
            regression_values = json.load(f)
        result = {}
        for case_id, value in regression_values.items():
            if isinstance(value, dict) and self.regression_key in value:
                result[case_id] = value[self.regression_key]
            elif isinstance(value, (int, float)):
                result[case_id] = value
            else:
                result[case_id] = 0.0
        return result

    def keys(self):
        return list(self.identifiers)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        item['regression_value'] = self.regression_values.get(key, 0.0)
        return item

    def load_case(self, key):
        data, seg, seg_prev, properties = super().load_case(key)
        reg_val = self.regression_values.get(key, 0.0)
        properties['regression_value'] = reg_val
        return data, seg, properties, reg_val


class RegnnUNetDataLoader(nnUNetDataLoaderBase):
    def __init__(self,
                 data: RegnnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        self._orig_data = data
        self.label_transform = RemoveLabelTansform(-1, 0)
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)

    def determine_shapes(self):
        k = self._data.identifiers[0]
        data, seg, _, _ = self._data.load_case(k)
        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.int16)
        regression_values = np.zeros((self.batch_size, 1), dtype=np.float32)
        properties_for_random_crop = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            case_data, case_seg, case_props, reg_val = self._data.load_case(i)
            case_seg = self._apply_label_transform(case_seg)
            shape = case_data.shape[1:]
            if not force_fg and not self.has_ignore:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, False, None)
            else:
                class_locations = case_props['class_locations']
                bbox_lbs, bbox_ubs = self.get_bbox(shape, True, class_locations)
            bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]
            data[j] = crop_and_pad_nd(case_data, bbox, 0)
            seg[j] = crop_and_pad_nd(case_seg, bbox, -1)
            regression_values[j, 0] = reg_val
            properties_for_random_crop.append(case_props)

        # transforms are handled by the training pipeline if provided

        return {
            'data': torch.from_numpy(data),
            'target': torch.from_numpy(seg),
            'seg': torch.from_numpy(seg),
            'properties': properties_for_random_crop,
            'keys': selected_keys,
            'regression_value': torch.from_numpy(regression_values)
        }

    def _apply_label_transform(self, seg: np.ndarray) -> np.ndarray:
        temp = {'data': None, 'seg': seg}
        self.label_transform(**temp)
        return temp['seg']

    # use crop_and_pad_nd from acvl_utils
