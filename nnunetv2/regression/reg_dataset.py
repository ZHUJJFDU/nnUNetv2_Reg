import os
import json
from typing import List, Dict, Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, isdir
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy as nnUNetDataset


class RegnnUNetDataset(nnUNetDataset):
    """
    Extension of nnUNetDataset that includes regression values.
    The regression values are loaded from a JSON file.
    """

    def __init__(self,
                 folder: str,
                 regression_values_file: str,
                 regression_key: str = "bulla_thickness",
                 case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        Initialize RegnnUNetDataset

        Args:
            folder: Path to the preprocessed dataset folder
            regression_values_file: Path to the JSON file containing regression values
            regression_key: Key in regression values JSON for the target value (default: "bulla_thickness")
            case_identifiers: List of case identifiers
            num_images_properties_loading_threshold: Threshold for loading properties
            folder_with_segs_from_previous_stage: Path to segmentations from previous stage
        """
        # Fix folder path if it contains duplicate directory names
        self.folder = folder

        # Check for duplicate directory structure
        folder_parts = os.path.normpath(folder).split(os.sep)
        if len(folder_parts) >= 2 and folder_parts[-1] == folder_parts[-2]:
            print(f"Detected duplicate directory structure: {folder}")
            # Use the parent directory instead
            fixed_folder = os.path.dirname(folder)
            print(f"Using fixed folder path: {fixed_folder}")
            folder = fixed_folder
            self.folder = fixed_folder

        # Initialize parent class
        super().__init__(folder, case_identifiers, num_images_properties_loading_threshold,
                         folder_with_segs_from_previous_stage)

        # Load regression values
        self.regression_values_file = regression_values_file
        self.regression_key = regression_key
        self.regression_values = self._load_regression_values()

        # Check if all cases have regression values
        missing_cases = []
        for case_id in self.dataset.keys():
            if case_id not in self.regression_values:
                missing_cases.append(case_id)

        if missing_cases:
            print(f"Warning: {len(missing_cases)} cases do not have regression values:")
            print(missing_cases[:10])
            if len(missing_cases) > 10:
                print(f"... and {len(missing_cases) - 10} more")

    def _load_regression_values(self):
        """
        Load regression values from JSON file

        Returns:
            Dictionary with regression values
        """
        if not isfile(self.regression_values_file):
            raise FileNotFoundError(f"Regression values file not found: {self.regression_values_file}")

        try:
            # 使用utf-8-sig编码来处理BOM
            with open(self.regression_values_file, 'r', encoding='utf-8-sig') as f:
                regression_values = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in regression values file {self.regression_values_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading regression values from {self.regression_values_file}: {e}")

        # Process the regression values based on their format
        result = {}

        # Check if we have a nested format: {"case_id": {"key": value}}
        for case_id, value in regression_values.items():
            if isinstance(value, dict) and self.regression_key in value:
                # Nested format
                result[case_id] = value[self.regression_key]
            elif isinstance(value, (int, float)):
                # Simple format: {"case_id": value}
                result[case_id] = value
            else:
                # Unknown format
                print(
                    f"Warning: Unsupported format for case {case_id}. Expected number or dict with key '{self.regression_key}'.")
                result[case_id] = 0.0

        return result

    def __getitem__(self, key):
        """
        Get an item from the dataset

        Args:
            key: Case identifier

        Returns:
            Dict with data, properties, and regression value
        """
        # Get the base item from parent class
        item = super().__getitem__(key)

        # Add regression value if available
        if key in self.regression_values:
            item['regression_value'] = self.regression_values[key]
        else:
            # Use a default value if not available
            print(f"Warning: No regression value found for case {key}. Using default value 0.0")
            item['regression_value'] = 0.0

        return item

    def load_case(self, key):
        """
        Load a case including its regression value

        Args:
            key: Case identifier

        Returns:
            Tuple of (data, seg, properties, regression_value)
            The regression value is also stored in properties['regression_value']
        """
        try:
            # Get data, seg, and properties from parent class
            data, seg, properties = super().load_case(key)

            # Get regression value
            regression_value = self.regression_values.get(key, 0.0)

            # Store regression value in properties
            properties['regression_value'] = regression_value

            # Return data, seg, properties, and regression_value
            return data, seg, properties, regression_value
        except FileNotFoundError as e:
            # If the file is not found, try to fix the path
            if "pkl" in str(e):
                print(f"Error loading case {key}: {str(e)}")
                print("Attempting to fix pkl file path...")

                # Try to find the pkl file in the parent directory
                parent_dir = os.path.dirname(self.dataset[key]['properties_file'])
                potential_pkl = join(parent_dir, f"{key}.pkl")

                if isfile(potential_pkl):
                    print(f"Found pkl file at: {potential_pkl}")
                    self.dataset[key]['properties_file'] = potential_pkl
                    return self.load_case(key)
                else:
                    # Try to find the pkl file in the grandparent directory
                    grandparent_dir = os.path.dirname(parent_dir)
                    potential_pkl = join(grandparent_dir, f"{key}.pkl")

                    if isfile(potential_pkl):
                        print(f"Found pkl file at: {potential_pkl}")
                        self.dataset[key]['properties_file'] = potential_pkl
                        return self.load_case(key)

                    # Try to find the pkl file in any subdirectory
                    for root, dirs, files in os.walk(os.path.dirname(grandparent_dir)):
                        if f"{key}.pkl" in files:
                            potential_pkl = join(root, f"{key}.pkl")
                            print(f"Found pkl file at: {potential_pkl}")
                            self.dataset[key]['properties_file'] = potential_pkl
                            return self.load_case(key)

            # If we couldn't fix it, raise the original error
            raise e