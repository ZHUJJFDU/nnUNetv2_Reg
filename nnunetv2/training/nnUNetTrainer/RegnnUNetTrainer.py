import os
import json
import torch
import time
import numpy as np
from typing import Union, Tuple, List, Dict
from torch.cuda.amp import autocast as cuda_autocast
import multiprocessing
import warnings
from time import sleep

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isfile, isdir, load_pickle, save_json
from torch import nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.reg_dataloader import RegnnUNetDataLoader
from nnunetv2.training.dataloading.reg_dataloader import RegnnUNetDataset as RegDataset
from nnunetv2.training.loss.reg_loss import DC_and_CE_and_Regression_loss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from regression_inference import RegPredictor as RegnnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


class RegnnUNetTrainer(nnUNetTrainer):
    """
    nnUNetTrainer extension that supports regression alongside segmentation
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        Initialize RegnnUNetTrainer
        
        Args:
            plans: nnUNet plans
            configuration: Configuration name (e.g., '3d_fullres')
            fold: Cross-validation fold
            dataset_json: Dataset JSON
            unpack_dataset: Whether to unpack the dataset
            device: Device to use for training
        """
        # Initialize regression-specific parameters with default values
        self.regression_values_file = None
        self.regression_key = 'reg'
        self.regression_weight = 0.1
        self.regression_loss_type = 'mse'
        self.use_regression = True
        self.debug = False
        
        # Initialize regression metrics
        self.regression_metrics = {
            'train_mse': [], 
            'train_mae': [], 
            'val_mse': [], 
            'val_mae': []
        }
        
        # Set default patch size to avoid AttributeError
        self.patch_size = None
        self.final_patch_size = None
        if plans is not None and configuration in plans['configurations']:
            if 'patch_size' in plans['configurations'][configuration]:
                self.patch_size = plans['configurations'][configuration]['patch_size']
                self.final_patch_size = self.patch_size  # 设置final_patch_size与patch_size相同
        
        # Set number of threads for multiprocessing
        self.num_threads_in_multithreaded = get_allowed_n_proc_DA()
        
        # 恢复AMP设置，不再强制关闭
        self.amp = True
        
        # Set deep supervision attribute
        self.deep_supervision = True
        
        # Call parent init
        super().__init__(plans, configuration, fold, dataset_json, device)

    def set_regression_parameters(self, reg_weight=1.0, reg_loss='mse', reg_key='reg', debug=True):
        """
        Set regression-specific parameters after initialization
        
        Args:
            reg_weight: Weight of the regression loss
            reg_loss: Regression loss to use ('mse' or 'mae')
            reg_key: Key for the regression value in the regression_values.json file
            debug: Whether to run in debug mode
        """
        self.regression_weight = reg_weight
        self.regression_loss_type = reg_loss
        self.regression_key = reg_key
        self.debug = debug
        self.print_to_log_file(f"Set regression parameters: weight={reg_weight}, loss={reg_loss}, key={reg_key}, debug={debug}")

    def initialize(self):
        """
        Initialize trainer with regression support
        """
        # Call parent initialize (builds default network)
        super().initialize()

        # Hardcode DualDecoderRegCBAMUNet for regression fine-tuning
        try:
            arch_class_name = "dynamic_network_architectures.architectures.dual_decoder_regression_cbamunet.DualDecoderRegCBAMUNet"
            arch_kwargs = dict(**self.configuration_manager.network_arch_init_kwargs)
            if 'n_blocks_per_stage' in arch_kwargs and 'n_conv_per_stage' not in arch_kwargs:
                arch_kwargs['n_conv_per_stage'] = arch_kwargs.pop('n_blocks_per_stage')
            arch_kwargs['regression_dim'] = 1
            arch_kwargs['enable_cross_attention'] = False
            req_import = self.configuration_manager.network_arch_init_kwargs_req_import
            new_net = get_network_from_plans(
                arch_class_name,
                arch_kwargs,
                req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                allow_init=True,
                deep_supervision=self.enable_deep_supervision
            ).to(self.device)
            self.network = new_net
            self.print_to_log_file("Switched to DualDecoderRegCBAMUNet for regression fine-tuning")
        except Exception as e:
            self.print_to_log_file(f"Failed to switch network architecture: {e}")

        # Knowledge Lock-in: freeze segmentation decoder and encoder to preserve segmentation and feature extraction knowledge
        try:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            frozen_seg = 0
            if hasattr(mod, 'seg_decoder') and hasattr(mod, 'reg_decoder'):
                for p in mod.seg_decoder.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_seg += 1
                self.print_to_log_file(f"Knowledge Lock-in: frozen {frozen_seg} parameters in seg_decoder")
            elif hasattr(mod, 'decoder'):
                for p in mod.decoder.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_seg += 1
                self.print_to_log_file(f"Knowledge Lock-in: frozen {frozen_seg} parameters in decoder")
            else:
                self.print_to_log_file("Knowledge Lock-in: no seg_decoder found, skip freezing")

            frozen_enc = 0
            if hasattr(mod, 'encoder'):
                for p in mod.encoder.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_enc += 1
                self.print_to_log_file(f"Knowledge Lock-in: frozen {frozen_enc} parameters in encoder")
            else:
                self.print_to_log_file("Knowledge Lock-in: no encoder found, skip freezing")

        except Exception as e:
            self.print_to_log_file(f"Knowledge Lock-in error: {e}")

        # Reconfigure optimizer to only include trainable parameters (exclude frozen modules)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        
        # Ensure amp and amp_grad_scaler are properly initialized
        if not hasattr(self, 'amp'):
            self.amp = True  # Enable AMP by default
        if not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()
        
        # Find the regression values file
        self._find_regression_values_file()
        
        # Build regression loss function
        self._build_regression_loss()
        
        # Print a message to confirm setup
        self.print_to_log_file(f"Regression trainer initialized with:")
        self.print_to_log_file(f"  - Regression values file: {self.regression_values_file}")
        self.print_to_log_file(f"  - Regression weight: {self.regression_weight}")
        self.print_to_log_file(f"  - Regression loss type: {self.regression_loss_type}")
        self.print_to_log_file(f"  - Regression key: {self.regression_key}")

        try:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            if hasattr(mod, 'encoder'):
                mod.encoder.eval()
            if hasattr(mod, 'seg_decoder'):
                mod.seg_decoder.eval()
            elif hasattr(mod, 'decoder'):
                mod.decoder.eval()
        except Exception:
            pass

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        try:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            if hasattr(mod, 'encoder'):
                mod.encoder.eval()
            if hasattr(mod, 'seg_decoder'):
                mod.seg_decoder.eval()
            elif hasattr(mod, 'decoder'):
                mod.decoder.eval()
        except Exception:
            pass

    def _find_regression_values_file(self):
        """
        Find the regression values file
        """
        dataset_folder = self.preprocessed_dataset_folder
        dataset_name = os.path.basename(dataset_folder)
        
        # Potential locations for regression_values.json
        potential_locations = [
            # Check in preprocessed dataset folder
            join(dataset_folder, "regression_values.json"),
            
            # Check in parent directory
            join(os.path.dirname(dataset_folder), "regression_values.json"),
            
            # Check in the raw dataset folder
            join(os.path.dirname(os.path.dirname(dataset_folder)), "nnUNet_raw", 
                  dataset_name, "regression_values.json"),
            
            # Check with just the dataset name in raw data
            join("DATASET", "nnUNet_raw", dataset_name, "regression_values.json"),
            
            # Check in the current working directory
            join(os.getcwd(), "regression_values.json"),
            
            # Check in a DATASET folder in current directory
            join(os.getcwd(), "DATASET", "nnUNet_raw", dataset_name, "regression_values.json"),
            
            # Additional locations for Dataset102_Reg
            join("DATASET", "nnUNet_raw", "Dataset102_Reg", "regression_values.json"),
            join(os.getcwd(), "DATASET", "nnUNet_raw", "Dataset102_Reg", "regression_values.json"),
            join(os.path.dirname(os.path.dirname(dataset_folder)), "nnUNet_raw", "Dataset102_Reg", "regression_values.json"),
            
            # Check in specific Dataset102_Reg folder
            join("DATASET", "Dataset102_Reg", "regression_values.json"),
            join(os.getcwd(), "DATASET", "Dataset102_Reg", "regression_values.json"),
            
            # Check in raw data folder
            join(os.getcwd(), "DATASET", "nnUNet_raw", "Dataset102_Reg", "regression_values.json")
        ]
        
        # Try each location
        for location in potential_locations:
            if isfile(location):
                self.regression_values_file = location
                self.print_to_log_file(f"Found regression values at: {self.regression_values_file}")
                return
        
        # If not found, search recursively
        self.print_to_log_file("Searching recursively for regression_values.json...")
        for root, dirs, files in os.walk(os.getcwd()):
            if "regression_values.json" in files:
                self.regression_values_file = join(root, "regression_values.json")
                self.print_to_log_file(f"Found regression values at: {self.regression_values_file}")
                return
        
        # If still not found, provide a clear error message
            err_msg = f"Regression values file (regression_values.json) not found in any of these locations:\n"
            for loc in potential_locations:
                err_msg += f"  - {loc}\n"
            err_msg += "Please ensure the file exists in one of these locations."
            raise FileNotFoundError(err_msg)
        
    def _build_regression_loss(self):
        """
        Build the combined loss function for segmentation and regression
        """
        # Create a new combined loss function using the DC_and_CE_and_Regression_loss
        soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        ce_kwargs = {'weight': None}
        
        # 增加回归权重，默认设为1.0而不是0.1
        regression_weight = 1.0 if self.regression_weight <= 0.1 else self.regression_weight
        
        self.loss = DC_and_CE_and_Regression_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            weight_reg=regression_weight,  # 使用更高的回归权重
            reg_loss_type=self.regression_loss_type,
            debug=True  # 启用调试模式以获取更多信息
        )
        
        # Move loss to device
        self.loss = self.loss.to(self.device)
        
        self.print_to_log_file(f"Using combined loss function: DC_and_CE_and_Regression_loss")
        self.print_to_log_file(f"  - Regression weight: {regression_weight}")
        self.print_to_log_file(f"  - Regression loss type: {self.regression_loss_type}")
        self.print_to_log_file(f"  - Debug mode enabled")

    def get_tr_and_val_datasets(self):
        """
        Create and return training and validation datasets with regression support
        """
        # Create the training dataset
        tr_keys, val_keys = self.do_split()
        
        # Make sure regression_values_file is set
        if self.regression_values_file is None:
            self._find_regression_values_file()
        
        main_data_dir = self.preprocessed_dataset_folder
        self.print_to_log_file(f"Main data directory: {main_data_dir}, exists: {isdir(main_data_dir)}")
        
        # Use the regression dataset with the main_data_dir (no tr/val folders)
        tr_dataset = RegDataset(
            main_data_dir,
            self.regression_values_file,
            self.regression_key,
            case_identifiers=tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        
        val_dataset = RegDataset(
            main_data_dir,
            self.regression_values_file,
            self.regression_key,
            case_identifiers=val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        
        self.print_to_log_file(f"Created training dataset with {len(tr_dataset.identifiers)} cases")
        self.print_to_log_file(f"Created validation dataset with {len(val_dataset.identifiers)} cases")
        
        return tr_dataset, val_dataset

    def get_dataloaders(self):
        """
        Create and return the data loaders for training and validation
        
        Returns:
            Tuple of (training data loader, validation data loader)
        """
        # Get datasets
        tr_dataset, val_dataset = self.get_tr_and_val_datasets()
        
        # Print dataset types for debugging
        self.print_to_log_file(f"Training dataset type: {type(tr_dataset).__name__}")
        self.print_to_log_file(f"Validation dataset type: {type(val_dataset).__name__}")
        
        # Create data loaders
        tr_loader = RegnnUNetDataLoader(
            tr_dataset,
            self.batch_size,
            self.patch_size,
            self.final_patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent
        )
        
        val_loader = RegnnUNetDataLoader(
            val_dataset,
            self.batch_size,
            self.patch_size,
            self.final_patch_size,
            self.label_manager,
            oversample_foreground_percent=0.0
        )
        
        # Use the MultiThreadedAugmenter like the parent class does
        from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        
        if self.num_threads_in_multithreaded > 1:
            mt_gen_train = MultiThreadedAugmenter(
                data_loader=tr_loader, 
                transform=None,
                num_processes=self.num_threads_in_multithreaded,
                pin_memory=self.device.type == 'cuda'
            )
            mt_gen_val = MultiThreadedAugmenter(
                data_loader=val_loader, 
                transform=None,
                num_processes=max(1, self.num_threads_in_multithreaded // 2),
                pin_memory=self.device.type == 'cuda'
            )
        else:
            mt_gen_train = SingleThreadedAugmenter(tr_loader, None)
            mt_gen_val = SingleThreadedAugmenter(val_loader, None)
            
        # Initialize the generators by fetching first batches
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        
        return mt_gen_train, mt_gen_val

    def configure_optimizers(self):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        params = [p for p in mod.parameters() if p.requires_grad]
        if len(params) == 0:
            self.print_to_log_file("WARNING: No trainable parameters found after Knowledge Lock-in; falling back to all parameters")
            params = list(mod.parameters())
        optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        """
        One training step
        
        Args:
            batch: Dictionary containing the batch data
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Get input data
        data = batch['data']
        target = batch['target']
        
        # Get regression target if available
        if 'regression_value' in batch:
            regression_target = batch['regression_value']
        else:
            # Use a dummy regression target if not available
            regression_target = torch.zeros((data.shape[0], 1), dtype=torch.float32)
        
        # Move data to device
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        regression_target = regression_target.to(self.device, non_blocking=True)
        
        # Reset optimizer
        self.optimizer.zero_grad()
        
        # Forward pass - 使用AMP
        if self.amp:
            with cuda_autocast():
                output = self.network(data)
                
                # Check output format and extract segmentation and regression outputs
                if isinstance(output, tuple) and len(output) == 2:
                    # Network returns (segmentation, regression)
                    seg_output, reg_output = output
                    # 确保回归输出的类型与回归目标匹配
                    if reg_output.dtype != regression_target.dtype:
                        reg_output = reg_output.to(dtype=regression_target.dtype)
                else:
                    # Network returns only segmentation
                    seg_output = output
                    reg_output = None
                
                # Handle case where seg_output is a list (deep supervision outputs)
                if isinstance(seg_output, list):
                    if self.deep_supervision:
                        loss = self.loss(seg_output, target, reg_output, regression_target)
                    else:
                        seg_output = seg_output[0]
                        loss = self.loss(seg_output, target, reg_output, regression_target)
                else:
                    loss = self.loss(seg_output, target, reg_output, regression_target)
                
                # Extract individual losses if returned
                if isinstance(loss, tuple) and len(loss) == 3:
                    total_loss, seg_loss, reg_loss = loss
                else:
                    total_loss = loss
                    seg_loss = loss
                    reg_loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass with AMP
            self.amp_grad_scaler.scale(total_loss).backward()
            self.amp_grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.amp_grad_scaler.step(self.optimizer)
            self.amp_grad_scaler.update()
        else:
            # 无AMP模式
            output = self.network(data)
            
            # Check output format and extract segmentation and regression outputs
            if isinstance(output, tuple) and len(output) == 2:
                # Network returns (segmentation, regression)
                seg_output, reg_output = output
            else:
                # Network returns only segmentation
                seg_output = output
                reg_output = None
            
            # Handle case where seg_output is a list (deep supervision outputs)
            if isinstance(seg_output, list):
                if self.deep_supervision:
                    loss = self.loss(seg_output, target, reg_output, regression_target)
                else:
                    seg_output = seg_output[0]
                    loss = self.loss(seg_output, target, reg_output, regression_target)
            else:
                loss = self.loss(seg_output, target, reg_output, regression_target)
            
            # Extract individual losses if returned
            if isinstance(loss, tuple) and len(loss) == 3:
                total_loss, seg_loss, reg_loss = loss
            else:
                total_loss = loss
                seg_loss = loss
                reg_loss = torch.tensor(0.0, device=self.device)
            
            # 标准反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            
        # Return loss values
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy() if hasattr(seg_loss, 'detach') else seg_loss,
            'reg_loss': reg_loss.detach().cpu().numpy() if hasattr(reg_loss, 'detach') else reg_loss
        }

    def validation_step(self, batch: dict) -> dict:
        """
        One validation step
        
        Args:
            batch: Dictionary containing the batch data
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Get input data
        data = batch['data']
        target = batch['target']
        
        # Get regression target if available
        if 'regression_value' in batch:
            regression_target = batch['regression_value']
        else:
            # Use a dummy regression target if not available
            regression_target = torch.zeros((data.shape[0], 1), dtype=torch.float32)
        
        # Move data to device
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        regression_target = regression_target.to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.no_grad():
            if self.amp:
                with cuda_autocast():
                    output = self.network(data)
                    
                    # Check output format and extract segmentation and regression outputs
                    if isinstance(output, tuple) and len(output) == 2:
                        # Network returns (segmentation, regression)
                        seg_output, reg_output = output
                        # 确保回归输出的类型与回归目标匹配
                        if reg_output.dtype != regression_target.dtype:
                            reg_output = reg_output.to(dtype=regression_target.dtype)
                    elif isinstance(output, tuple) and len(output) == 1:
                        # Network returns tuple with single element (some networks do this)
                        seg_output = output[0]
                        reg_output = None
                    else:
                        # Network returns only segmentation
                        seg_output = output
                        reg_output = None
                    
                    # Handle case where seg_output is a list (deep supervision outputs)
                    if isinstance(seg_output, list):
                        if self.deep_supervision:
                            loss = self.loss(seg_output, target, reg_output, regression_target)
                        else:
                            seg_output = seg_output[0]
                            loss = self.loss(seg_output, target, reg_output, regression_target)
                    else:
                        loss = self.loss(seg_output, target, reg_output, regression_target)
            else:
                output = self.network(data)
                
                # Check output format and extract segmentation and regression outputs
                if isinstance(output, tuple) and len(output) == 2:
                    # Network returns (segmentation, regression)
                    seg_output, reg_output = output
                elif isinstance(output, tuple) and len(output) == 1:
                    # Network returns tuple with single element (some networks do this)
                    seg_output = output[0]
                    reg_output = None
                else:
                    # Network returns only segmentation
                    seg_output = output
                    reg_output = None
                    
                # Handle case where seg_output is a list (deep supervision outputs)
                if isinstance(seg_output, list):
                    if self.deep_supervision:
                        loss = self.loss(seg_output, target, reg_output, regression_target)
                    else:
                        seg_output = seg_output[0]
                        loss = self.loss(seg_output, target, reg_output, regression_target)
                else:
                    loss = self.loss(seg_output, target, reg_output, regression_target)
            
            # Extract individual losses if returned
            if isinstance(loss, tuple) and len(loss) == 3:
                total_loss, seg_loss, reg_loss = loss
            else:
                total_loss = loss
                seg_loss = loss
                reg_loss = torch.tensor(0.0, device=self.device)
                
            # Calculate regression metrics if available
            if reg_output is not None and regression_target is not None:
                # Ensure regression target has the right shape
                if regression_target.ndim == 1 and reg_output.ndim > 1:
                    regression_target = regression_target.view(-1, 1)
        
                # Calculate MSE and MAE
                mse = torch.nn.functional.mse_loss(reg_output, regression_target)
                mae = torch.nn.functional.l1_loss(reg_output, regression_target)
            else:
                mse = torch.tensor(0.0, device=self.device)
                mae = torch.tensor(0.0, device=self.device)
            
            # Calculate segmentation metrics (tp, fp, fn)
            if isinstance(seg_output, list):
                pred = seg_output[0]
            else:
                pred = seg_output
            
            # 修复：确保pred是张量而不是元组
            if isinstance(pred, tuple):
                # 如果网络直接返回了元组，取第一个元素作为分割输出
                pred = pred[0]
                
            pred_softmax = torch.softmax(pred, dim=1)
            pred_argmax = pred_softmax.argmax(dim=1)
            
            # Convert target from one-hot to single channel if needed
            if target.shape[1] > 1:
                target_classes = target.argmax(dim=1)
            else:
                target_classes = target[:, 0].long()
            
            # Calculate true positives, false positives, and false negatives
            tp_hard = torch.zeros((target.shape[0], self.label_manager.num_segmentation_heads), device=self.device)
            fp_hard = torch.zeros((target.shape[0], self.label_manager.num_segmentation_heads), device=self.device)
            fn_hard = torch.zeros((target.shape[0], self.label_manager.num_segmentation_heads), device=self.device)
            
            for b in range(target.shape[0]):
                for c in range(1, self.label_manager.num_segmentation_heads):
                    # True positives: pixels where both prediction and target are class c
                    tp_hard[b, c] = torch.sum((pred_argmax[b] == c) & (target_classes[b] == c)).float()
                    # False positives: pixels where prediction is class c but target is not
                    fp_hard[b, c] = torch.sum((pred_argmax[b] == c) & (target_classes[b] != c)).float()
                    # False negatives: pixels where target is class c but prediction is not
                    fn_hard[b, c] = torch.sum((pred_argmax[b] != c) & (target_classes[b] == c)).float()
        
        # Return metrics
        return {
            'loss': total_loss.cpu().numpy(),
            'seg_loss': seg_loss.cpu().numpy() if hasattr(seg_loss, 'cpu') else seg_loss,
            'reg_loss': reg_loss.cpu().numpy() if hasattr(reg_loss, 'cpu') else reg_loss,
            'mse': mse.cpu().numpy(),
            'mae': mae.cpu().numpy(),
            'tp_hard': tp_hard.cpu().numpy(),
            'fp_hard': fp_hard.cpu().numpy(),
            'fn_hard': fn_hard.cpu().numpy()
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        Called at the end of each training epoch
        
        Args:
            train_outputs: List of outputs from train_step
        """
        # Call parent method
        super().on_train_epoch_end(train_outputs)
        
        # Update regression metrics
        if train_outputs:
            reg_losses = [o['reg_loss'] for o in train_outputs if 'reg_loss' in o]
            if reg_losses:
                self.regression_metrics['train_mse'].append(np.mean(reg_losses))
                self.print_to_log_file(f"Epoch {self.current_epoch} train regression loss: {np.mean(reg_losses):.4f}")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Called at the end of each validation epoch
        
        Args:
            val_outputs: List of outputs from validation_step
        """
        # Call parent method
        super().on_validation_epoch_end(val_outputs)
        
        # Update regression metrics
        if val_outputs:
            mse_values = [o['mse'] for o in val_outputs if 'mse' in o]
            mae_values = [o['mae'] for o in val_outputs if 'mae' in o]
            
            if mse_values:
                self.regression_metrics['val_mse'].append(np.mean(mse_values))
                self.regression_metrics['val_mae'].append(np.mean(mae_values))
                self.print_to_log_file(f"Epoch {self.current_epoch} val regression - MSE: {np.mean(mse_values):.4f}, MAE: {np.mean(mae_values):.4f}")

    def save_checkpoint(self, filename: str) -> None:
        """
        Save checkpoint with regression metrics
        
        Args:
            filename: Path to save the checkpoint
        """
        # Call parent method to save network weights
        super().save_checkpoint(filename)
        
        # Convert NumPy values to regular Python types for JSON serialization
        metrics_to_save = {}
        for metric_key, metric_values in self.regression_metrics.items():
            if isinstance(metric_values, list):
                # Convert each item in the list
                metrics_to_save[metric_key] = [float(x) if hasattr(x, 'item') else x for x in metric_values]
            else:
                # Convert the metric value itself
                metrics_to_save[metric_key] = float(metric_values) if hasattr(metric_values, 'item') else metric_values
        
        # Save regression metrics
        metrics_file = filename.replace('.pth', '_regression_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        self.print_to_log_file(f"Saved regression metrics to {metrics_file}")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        Partial load for regression fine-tuning: only encoder and segmentation decoder weights
        """
        if not self.was_initialized:
            self.initialize()
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
            metrics_file = filename_or_checkpoint.replace('.pth', '_regression_metrics.json')
        else:
            checkpoint = filename_or_checkpoint
            metrics_file = None
        state = checkpoint['network_weights'] if 'network_weights' in checkpoint else checkpoint
        filtered = {}
        for k, v in state.items():
            key = k[7:] if k.startswith('module.') else k
            if key.startswith('encoder.') or key.startswith('decoder.'):
                filtered[key] = v
        missing, unexpected = self.network.load_state_dict(filtered, strict=False)
        self.print_to_log_file(f"Loaded encoder+decoder weights with {len(missing)} missing, {len(unexpected)} unexpected keys")
        if metrics_file and isfile(metrics_file):
            with open(metrics_file, 'r') as f:
                self.regression_metrics = json.load(f)
            self.print_to_log_file(f"Loaded regression metrics from {metrics_file}")

    def _convert_to_npy(self, npz_file, unpack_segmentation=True, overwrite_existing=False, verify_npy=False):
        """
        Convert npz files to npy files using the utility function from utils.py
        
        Args:
            npz_file: Path to the npz file
            unpack_segmentation: Whether to unpack segmentation data
            overwrite_existing: Whether to overwrite existing npy files
            verify_npy: Whether to verify the created npy files
            
        Returns:
            None
        """
        from nnunetv2.training.dataloading.utils import _convert_to_npy as convert_func
        self.print_to_log_file(f"Converting {npz_file} to npy format")
        convert_func(npz_file, unpack_segmentation, overwrite_existing, verify_npy)

    def batch_convert_to_npy(self, folder, unpack_segmentation=True, overwrite_existing=False, verify_npy=False, num_processes=None):
        """
        Convert all npz files in a folder to npy files
        
        Args:
            folder: Path to the folder containing npz files
            unpack_segmentation: Whether to unpack segmentation data
            overwrite_existing: Whether to overwrite existing npy files
            verify_npy: Whether to verify the created npy files
            num_processes: Number of processes to use
            
        Returns:
            None
        """
        from nnunetv2.training.dataloading.utils import unpack_dataset
        self.print_to_log_file(f"Converting all npz files in {folder} to npy format")
        
        # Use the utility function from utils.py
        if num_processes is None:
            from nnunetv2.configuration import default_num_processes
            num_processes = default_num_processes
            
        unpack_dataset(folder, unpack_segmentation, overwrite_existing, num_processes, verify_npy)
        self.print_to_log_file(f"Finished converting npz files in {folder} to npy format")

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        重写验证方法，使用RegnnUNetPredictor处理回归输出
        """
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        # 使用RegnnUNetPredictor替代原始的nnUNetPredictor
        predictor = RegnnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                      self.dataset_json, self.__class__.__name__,
                                      self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # 创建回归指标输出文件
            regression_metrics_file = join(self.output_folder, 'checkpoint_final_regression_metrics.json')
            regression_predictions = {}

            # 我们不能使用self.get_tr_and_val_datasets()，因为我们可能是DDP，然后我们必须在workers之间分配验证keys
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # 我们不能到处设置障碍，因为每个GPU接收的keys数量可能不同

            dataset_val = RegDataset(
                self.preprocessed_dataset_folder,
                self.regression_values_file,
                self.regression_key,
                case_identifiers=val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                         allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                             allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties, _ = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                      output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                try:
                    # 使用RegnnUNetPredictor预测，可能返回回归值
                    prediction_result = predictor.predict_sliding_window_return_logits(data)
                    
                    # 检查是否有回归输出
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        prediction, regression = prediction_result
                        prediction = prediction.cpu()
                        regression = regression.cpu()
                        
                        # 保存回归预测结果
                        regression_value = regression.numpy().mean()  # 取平均值作为最终预测
                        regression_predictions[k] = float(regression_value)
                        
                        self.print_to_log_file(f"Case {k} regression prediction: {regression_value:.4f}")
                    else:
                        prediction = prediction_result
                        prediction = prediction.cpu()
                except RuntimeError as e:
                    self.print_to_log_file(f"Error during prediction of case {k}: {e}")
                    continue

                # 这需要进入后台进程
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                # 如果需要，导出下一阶段的softmax预测
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                          next_stage_config_manager.data_identifier)

                        try:
                            # 我们这样做，以便可以使用load_case而不必硬编码如何加载训练案例
                            tmp = RegDataset(
                                expected_preprocessed_folder,
                                self.regression_values_file,
                                self.regression_key,
                                case_identifiers=[k]
                            )
                            d, s, p, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))
                # 如果我们不时地设置障碍，对于大型数据集，我们将获得nccl超时。呸。
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            # 计算分割指标
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                              validation_output_folder,
                                              join(validation_output_folder, 'summary.json'),
                                              self.plans_manager.image_reader_writer_class(),
                                              self.dataset_json["file_ending"],
                                              self.label_manager.foreground_regions if self.label_manager.has_regions else
                                              self.label_manager.foreground_labels,
                                              self.label_manager.ignore_label, chill=True,
                                              num_processes=default_num_processes * dist.get_world_size() if
                                              self.is_ddp else default_num_processes)
            
            # 保存回归指标
            if regression_predictions:
                # 加载真实回归值（从properties或其他地方）
                # 这里假设我们有一个方法来获取真实回归值
                regression_gt = self._get_regression_ground_truth(val_keys)
                
                # 计算回归指标
                if regression_gt:
                    mse_values = []
                    mae_values = []
                    
                    for k in regression_predictions:
                        if k in regression_gt:
                            pred = regression_predictions[k]
                            gt = regression_gt[k]
                            mse_values.append((pred - gt) ** 2)
                            mae_values.append(abs(pred - gt))
                    
                    if mse_values:
                        mean_mse = sum(mse_values) / len(mse_values)
                        mean_mae = sum(mae_values) / len(mae_values)
                        
                        regression_metrics = {
                            'mse': mean_mse,
                            'mae': mean_mae,
                            'predictions': regression_predictions,
                            'ground_truth': regression_gt
                        }
                        
                        # 保存回归指标
                        save_json(regression_metrics, regression_metrics_file)
                        self.print_to_log_file(f"Saved regression metrics to {regression_metrics_file}")
            
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                 also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def _get_regression_ground_truth(self, val_keys):
        """
        获取验证集的回归真实值
        
        Args:
            val_keys: 验证集的keys
            
        Returns:
            包含回归真实值的字典 {case_id: regression_value}
        """
        # 初始化结果字典
        regression_gt = {}
        
        # 创建临时数据集对象用于加载验证集数据
        dataset_val = RegnnUNetDataset(
            self.preprocessed_dataset_folder,
            self.regression_values_file,
            self.regression_key,
            case_identifiers=val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        
        # 从数据集中加载回归值
        # 这里假设回归值存储在properties中的'regression_value'键下
        for k in val_keys:
            try:
                # 尝试从验证集加载回归值
                _, _, properties, _ = dataset_val.load_case(k)
                if 'regression_value' in properties:
                    regression_gt[k] = float(properties['regression_value'])
            except Exception as e:
                self.print_to_log_file(f"Error loading regression ground truth for case {k}: {e}")
        
        return regression_gt
