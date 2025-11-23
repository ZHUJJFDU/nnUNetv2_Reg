import argparse
import os
import itertools
import json
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from queue import Queue
from threading import Thread
from tqdm import tqdm
import multiprocessing
from time import sleep
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.inference.export_prediction import export_prediction_from_logits, convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


class RegPredictor(nnUNetPredictor):
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This function is overwritten from nnUNetPredictor because we need to manually specify the network architecture.
        The default nnUNetPredictor loads the architecture from the plans file, which is not correct for this
        regression model.
        """
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        if not use_folds:
            checkpoint = torch.load(join(model_training_output_dir, checkpoint_name), map_location=torch.device('cpu'), weights_only=False)
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes')
            parameters.append(checkpoint['network_weights'])
        else:
            for i, f in enumerate(use_folds):
                f = int(f) if f != 'all' else f
                checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                        map_location=torch.device('cpu'), weights_only=False)
                if i == 0:
                    trainer_name = checkpoint['trainer_name']
                    configuration_name = checkpoint['init_args']['configuration']
                    inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes')
                parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)

        # Manually build DualDecoderRegCBAMUNet for inference
        print("Manually building DualDecoderRegCBAMUNet for inference.")
        
        arch_class_name = "dynamic_network_architectures.architectures.dual_decoder_regression_cbamunet.DualDecoderRegCBAMUNet"
        arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
        if 'n_blocks_per_stage' in arch_kwargs and 'n_conv_per_stage' not in arch_kwargs:
            arch_kwargs['n_conv_per_stage'] = arch_kwargs.pop('n_blocks_per_stage')
        arch_kwargs['regression_dim'] = 1
        arch_kwargs['enable_cross_attention'] = False
        req_import = configuration_manager.network_arch_init_kwargs_req_import
        
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        
        network = get_network_from_plans(
            arch_class_name,
            arch_kwargs,
            req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            allow_init=True,
            deep_supervision=False
        )
        
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        
        if len(self.list_of_parameters) > 1:
            print("WARNING: More than one set of parameters found. Using only the first one.")
        
        self.network.load_state_dict(self.list_of_parameters[0])
        
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if isinstance(prediction, list):
            prediction = prediction[0]
        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)]
            for axes in axes_combinations:
                p = self.network(torch.flip(x, axes))
                if isinstance(p, tuple):
                    p = p[0]
                if isinstance(p, list):
                    p = p[0]
                prediction += torch.flip(p, axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def predict_from_files_with_regression(self,
                                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                           output_folder_or_list_of_truncated_output_files: Union[str, List[str]],
                                           save_probabilities: bool = False,
                                           overwrite: bool = True,
                                           num_processes_preprocessing: int = 8,
                                           num_processes_segmentation_export: int = 8,
                                           part_id: int = 0,
                                           num_parts: int = 1):
        if self.verbose:
            print(f"Predicting with part_id {part_id} of {num_parts} (max process ID is {num_parts - 1})")

        self._current_part_id = part_id
        self._current_num_parts = num_parts
        if self.verbose:
            print("\nRunning segmentation+regression prediction and saving results.")
        super().predict_from_files(list_of_lists_or_source_folder,
                                   output_folder_or_list_of_truncated_output_files,
                                   save_probabilities,
                                   overwrite,
                                   num_processes_preprocessing,
                                   num_processes_segmentation_export,
                                   folder_with_segs_from_prev_stage=None,
                                   num_parts=num_parts,
                                   part_id=part_id)

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None
        if output_folder is not None and part_id == num_parts - 1:
            regression_output_folder = join(output_folder, "regression_predictions")
            self.merge_regression_results(regression_output_folder, num_parts)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 8):
        regression_results = {}
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            output_folder = None
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f"\nPredicting {os.path.basename(ofile)}:")
                    output_folder = os.path.dirname(ofile)
                else:
                    print(f"\nPredicting image of shape {data.shape}:")
                print(f'perform_everything_on_device: {self.perform_everything_on_device}')
                properties = preprocessed['data_properties']
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                case_id = os.path.basename(ofile) if ofile is not None else 'case'
                case_id = os.path.splitext(case_id)[0]
                if hasattr(self, '_last_regression_value') and self._last_regression_value is not None:
                    regression_results[case_id] = float(self._last_regression_value)
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f"\nDone with image of shape {data.shape}:")
            ret = [i.get()[0] for i in r]
        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()
        compute_gaussian.cache_clear()
        empty_cache(self.device)
        if output_folder is not None:
            regression_output_folder = join(output_folder, "regression_predictions")
            maybe_mkdir_p(regression_output_folder)
            regression_json_path = join(regression_output_folder, f"regression_results_part_{getattr(self, '_current_part_id', 0)}.json")
            with open(regression_json_path, 'w') as f:
                json.dump(regression_results, f, indent=4)
            if self.verbose:
                print(f"Regression results saved to: {regression_json_path}")
        return ret

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = gaussian = None
        results_device = self.device if do_on_device else torch.device('cpu')
        try:
            empty_cache(self.device)
            data = data.to(results_device)
            if self.verbose:
                print(f'move image to device {results_device}')
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1
            reg_sum = torch.zeros(1, device=results_device, dtype=torch.float32)
            reg_weight = torch.zeros(1, device=results_device, dtype=torch.float32)
            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                for sl in slicers:
                    workon = torch.clone(data[sl][None], memory_format=torch.contiguous_format).to(self.device)
                    mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
                    if mirror_axes is not None:
                        assert max(mirror_axes) <= workon.ndim - 3
                        axes_shifted = [m + 2 for m in mirror_axes]
                        axes_combinations = [c for i in range(len(axes_shifted)) for c in itertools.combinations(axes_shifted, i + 1)]
                        seg_pred, reg_pred = None, None
                        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                            out = self.network(workon)
                            s = out[0] if isinstance(out, (tuple, list)) else out
                            r = out[1] if isinstance(out, (tuple, list)) else None
                            seg_pred = s
                            reg_pred = r
                            for axes in axes_combinations:
                                out_m = self.network(torch.flip(workon, axes))
                                s_m = out_m[0] if isinstance(out_m, (tuple, list)) else out_m
                                r_m = out_m[1] if isinstance(out_m, (tuple, list)) else None
                                seg_pred = seg_pred + torch.flip(s_m, axes)
                                if reg_pred is not None:
                                    reg_pred = reg_pred + r_m
                            seg_pred = seg_pred / (len(axes_combinations) + 1)
                            if reg_pred is not None:
                                reg_pred = reg_pred / (len(axes_combinations) + 1)
                    else:
                        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                            out = self.network(workon)
                            seg_pred = out[0] if isinstance(out, (tuple, list)) else out
                            reg_pred = out[1] if isinstance(out, (tuple, list)) else None
                    # remove batch dim, ensure shape [C, patch]
                    if isinstance(seg_pred, list):
                        seg_pred = seg_pred[0]
                    if seg_pred.shape[0] == 1:
                        seg_pred = seg_pred[0]
                    seg_pred = seg_pred.to(results_device)
                    predicted_logits[sl] += (seg_pred * gaussian)
                    n_predictions[sl[1:]] += gaussian
                    if reg_pred is not None:
                        w = gaussian.mean().to(torch.float32) if isinstance(gaussian, torch.Tensor) else torch.tensor(float(gaussian), device=results_device)
                        reg_sum += reg_pred.squeeze().to(results_device, dtype=torch.float32) * w
                        reg_weight += w
                    pbar.update()
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if reg_weight.item() > 0:
                self._last_regression_value = float((reg_sum / reg_weight).item())
            else:
                self._last_regression_value = None
        except Exception as e:
            del predicted_logits, n_predictions, gaussian
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def get_data_iterator_from_lists_of_lists(self,
                                              list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                              output_folder_or_list_of_truncated_output_files: Union[str, List[str]],
                                              overwrite: bool,
                                              part_id: int,
                                              num_parts: int,
                                              save_probabilities: bool,
                                              num_processes_preprocessing: int):
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        managed = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder,
            folder_with_segs_from_prev_stage=None,
            overwrite=overwrite,
            part_id=part_id,
            num_parts=num_parts,
            save_probabilities=save_probabilities
        )
        filtered_list_of_lists, output_filename_truncated, seg_from_prev_stage_files = managed
        return self._internal_get_data_iterator_from_lists_of_filenames(
            filtered_list_of_lists,
            seg_from_prev_stage_files,
            output_filename_truncated,
            num_processes_preprocessing
        )

    def data_iterator_to_torch(self, iterator):
        for preprocessed in iterator:
            yield preprocessed

    def merge_regression_results(self, folder, num_parts):
        all_results = {}
        for i in range(num_parts):
            json_path = join(folder, f"regression_results_part_{i}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    all_results.update(json.load(f))
                os.remove(json_path)  # Clean up partial files

        final_json_path = join(folder, "regression_results_all.json")
        with open(final_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        if self.verbose:
            print(f"All regression results merged and saved to: {final_json_path}")


def main():
    parser = argparse.ArgumentParser(description='Regression model inference script.')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input folder containing the images for prediction.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output folder for the segmentation and regression results.')
    parser.add_argument('-d', '--dataset_name_or_id', type=str, required=True,
                        help='Dataset name or ID to identify the trained model.')
    parser.add_argument('-tr', '--trainer_name', type=str, default='RegnnUNetTrainer',
                        help='Name of the trainer class used for training.')
    parser.add_argument('-p', '--plans_identifier', type=str, default='nnUNetPlans',
                        help='Identifier for the plans file.')
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres',
                        help='Configuration used for training.')
    parser.add_argument('-f', '--folds', nargs='+', type=int, default=None,
                        help='List of folds to use for inference. Default is all folds.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth',
                        help='Name of the checkpoint file to use.')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size for sliding window prediction.')
    parser.add_argument('--disable_tta', action='store_true',
                        help='Disable test-time augmentation.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference (e.g., "cuda" or "cpu").')
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Overwrite existing predictions.')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save softmax probabilities.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output.')
    parser.add_argument('--num_processes_preprocessing', type=int, default=8,
                        help='Number of processes for preprocessing.')
    parser.add_argument('--num_processes_segmentation_export', type=int, default=8,
                        help='Number of processes for segmentation export.')
    args = parser.parse_args()

    # Print inference parameters
    if not args.quiet:
        print("Regression model inference parameters:")
        for arg, value in sorted(vars(args).items()):
            print(f"  {arg}: {value}")
        print()

    # Auto-build model path
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    model_folder = join(nnUNet_results, dataset_name, f"{args.trainer_name}__{args.plans_identifier}__{args.configuration}")
    if not isdir(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    # Initialize the predictor
    predictor = RegPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=not args.quiet,
        allow_tqdm=not args.quiet
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=args.folds,
        checkpoint_name=args.checkpoint
    )

    # Run prediction
    predictor.predict_from_files_with_regression(
        args.input,
        args.output,
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=args.num_processes_preprocessing,
        num_processes_segmentation_export=args.num_processes_segmentation_export
    )


if __name__ == '__main__':
    main()
