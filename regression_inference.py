import os
import argparse
import torch
import numpy as np
from typing import Union, Tuple, List
import itertools
from contextlib import contextmanager
from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, maybe_mkdir_p, subfiles, save_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from acvl_utils.cropping_and_padding.padding import pad_nd_image


@contextmanager
def dummy_context():
    yield None


class RegPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_regression = kwargs.pop('return_regression', True)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)
        if isinstance(prediction, tuple) and len(prediction) == 2:
            seg_output, reg_output = prediction
            has_regression = True
        else:
            seg_output = prediction
            reg_output = None
            has_regression = False
        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)]
            for axes in axes_combinations:
                flipped_x = torch.flip(x, axes)
                mirrored_prediction = self.network(flipped_x)
                if isinstance(mirrored_prediction, tuple) and len(mirrored_prediction) == 2:
                    mirrored_seg, mirrored_reg = mirrored_prediction
                    mirrored_seg = torch.flip(mirrored_seg, axes)
                    if has_regression:
                        seg_output += mirrored_seg
                        reg_output += mirrored_reg
                    else:
                        seg_output += mirrored_seg
                        has_regression = True
                        reg_output = mirrored_reg
                else:
                    mirrored_seg = torch.flip(mirrored_prediction, axes)
                    seg_output += mirrored_seg
            seg_output /= (len(axes_combinations) + 1)
            if has_regression and reg_output is not None:
                reg_output /= (len(axes_combinations) + 1)
        if has_regression and reg_output is not None:
            return seg_output, reg_output
        else:
            return seg_output

    def _internal_predict_sliding_window_return_logits(self, data: torch.Tensor, slicers, do_on_device: bool = True):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        predicted_regression = None
        results_device = self.device if do_on_device else torch.device('cpu')
        has_regression_output = False
        try:
            empty_cache(self.device)
            data = data.to(results_device)
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.half, device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8, value_scaling_factor=10, device=results_device) if self.use_gaussian else 1
            regression_predictions = []
            for sl in slicers:
                workon = data[sl][None]
                workon = workon.to(self.device)
                prediction_result = self._internal_maybe_mirror_and_predict(workon)
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    prediction, regression = prediction_result
                    regression = regression.to('cpu')
                    regression_predictions.append(regression.detach())
                    has_regression_output = True
                else:
                    prediction = prediction_result
                prediction = prediction.to(results_device)
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian
            predicted_logits /= n_predictions
            if has_regression_output and len(regression_predictions) > 0:
                predicted_regression = torch.mean(torch.cat(regression_predictions, dim=0), dim=0, keepdim=True)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array.')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            if has_regression_output and len(regression_predictions) > 0:
                del regression_predictions
                if predicted_regression is not None:
                    del predicted_regression
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        if has_regression_output and predicted_regression is not None:
            return predicted_logits, predicted_regression
        else:
            return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                if self.perform_everything_on_device and self.device != 'cpu':
                    try:
                        prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                    except RuntimeError:
                        empty_cache(self.device)
                        prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                empty_cache(self.device)
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    predicted_logits, predicted_regression = prediction_result
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                    return predicted_logits, predicted_regression
                else:
                    predicted_logits = prediction_result
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                    return predicted_logits


def run_regression_prediction(input_folder: str,
                              output_folder: str,
                              model_folder: str,
                              use_folds: Union[List[int], Tuple[int, ...]] = None,
                              tile_step_size: float = 0.5,
                              use_gaussian: bool = True,
                              use_mirroring: bool = True,
                              perform_everything_on_gpu: bool = True,
                              verbose: bool = True,
                              save_probabilities: bool = False,
                              overwrite: bool = True,
                              checkpoint_name: str = "checkpoint_final.pth",
                              num_threads_preprocessing: int = 8,
                              num_threads_nifti_save: int = 2):
    maybe_mkdir_p(output_folder)
    device = torch.device("cuda" if torch.cuda.is_available() and perform_everything_on_gpu else "cpu")
    predictor = RegPredictor(tile_step_size=tile_step_size,
                             use_gaussian=use_gaussian,
                             use_mirroring=use_mirroring,
                             perform_everything_on_device=perform_everything_on_gpu,
                             device=device,
                             verbose=verbose,
                             verbose_preprocessing=verbose,
                             allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds, checkpoint_name)
    file_ending = predictor.dataset_json.get("file_ending", ".nii.gz")
    input_files = subfiles(input_folder, suffix=file_ending, join=True)
    input_files = [os.path.normpath(i) for i in input_files]
    output_files = []
    for i in input_files:
        fn = os.path.basename(i)
        if fn.endswith(file_ending):
            fn = fn[:-len(file_ending)]
        output_files.append(os.path.normpath(join(output_folder, fn)))
    predictor.predict_from_files(input_files, output_files, save_probabilities, overwrite,
                                 num_threads_preprocessing, num_threads_nifti_save)
    regression_results = {}
    for inp in input_files:
        img = predictor.preprocess_input_file(inp)
        with torch.no_grad():
            res = predictor.predict_sliding_window_return_logits(img)
            if isinstance(res, tuple) and len(res) == 2:
                _, reg = res
                regression_results[os.path.basename(inp)] = float(reg.cpu().numpy().mean())
    save_json(regression_results, os.path.normpath(join(output_folder, "regression_results.json")))
    return regression_results


def main():
    parser = argparse.ArgumentParser(description="回归模型推理")
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-m", "--model_folder", type=str, required=True)
    parser.add_argument("-f", "--folds", nargs="+", type=int, default=None)
    parser.add_argument("-s", "--step_size", type=float, default=0.5)
    parser.add_argument("--no_gaussian", action="store_true")
    parser.add_argument("--no_mirroring", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--save_probabilities", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument("--threads_preprocessing", type=int, default=8)
    parser.add_argument("--threads_save", type=int, default=2)
    args = parser.parse_args()
    run_regression_prediction(
        args.input_folder,
        args.output_folder,
        args.model_folder,
        args.folds,
        args.step_size,
        not args.no_gaussian,
        not args.no_mirroring,
        not args.cpu,
        args.verbose,
        args.save_probabilities,
        args.overwrite,
        args.checkpoint,
        args.threads_preprocessing,
        args.threads_save
    )


if __name__ == "__main__":
    main()
