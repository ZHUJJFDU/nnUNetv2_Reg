import multiprocessing
import os
import socket
from typing import Union, Optional

import sys
from pathlib import Path
# prefer local nnUNet repo over site-packages
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
import nnunetv2
import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def find_free_network_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'SemiSupervisedTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          device: torch.device = torch.device('cuda')):
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ({join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}).')
    assert issubclass(nnunet_trainer, nnUNetTrainer)

    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three task ID digits. Your input: {dataset_name_or_id}')

    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print("WARNING: Cannot continue training because there seems to be no checkpoint available. Starting new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError("Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def _apply_semi_params(trainer: nnUNetTrainer,
                       unlabeled_data_path: Optional[str],
                       unlabeled_batch_size: int,
                       consistency_weight: float,
                       ema_decay: float,
                       consistency_ramp_up_epochs: int,
                       save_interval: int,
                       use_confidence_mask: bool,
                       confidence_threshold: float,
                       use_entropy_filtering: bool,
                       entropy_threshold: float):
    if hasattr(trainer, 'ema_decay'):
        trainer.ema_decay = ema_decay
    if hasattr(trainer, 'consistency_weight'):
        trainer.consistency_weight = consistency_weight
    if hasattr(trainer, 'consistency_ramp_up_epochs'):
        trainer.consistency_ramp_up_epochs = consistency_ramp_up_epochs
    if hasattr(trainer, 'unlabeled_batch_size'):
        trainer.unlabeled_batch_size = unlabeled_batch_size
    if hasattr(trainer, 'unlabeled_data_path'):
        trainer.unlabeled_data_path = unlabeled_data_path
    trainer.save_every = save_interval
    if hasattr(trainer, 'use_confidence_mask'):
        trainer.use_confidence_mask = use_confidence_mask
    if hasattr(trainer, 'confidence_threshold'):
        trainer.confidence_threshold = confidence_threshold
    if hasattr(trainer, 'use_entropy_filter'):
        trainer.use_entropy_filter = use_entropy_filtering
    if hasattr(trainer, 'entropy_threshold'):
        trainer.entropy_threshold = entropy_threshold


def run_ddp(rank,
            dataset_name_or_id,
            configuration,
            fold,
            tr,
            p,
            disable_checkpointing,
            c,
            val,
            teacher_checkpoint,
            world_size,
            semi_kwargs):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    _apply_semi_params(nnunet_trainer, **semi_kwargs)

    assert not (c and val)

    maybe_load_checkpoint(nnunet_trainer, c, val, teacher_checkpoint)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    nnunet_trainer.perform_actual_validation(False)
    cleanup_ddp()


def run_semi_training(dataset_name_or_id: Union[str, int],
                      configuration: str,
                      fold: Union[int, str],
                      trainer_class_name: str = 'SemiSupervisedTrainer',
                      plans_identifier: str = 'nnUNetPlans',
                      teacher_checkpoint: Optional[str] = None,
                      unlabeled_data_path: Optional[str] = None,
                      unlabeled_batch_size: int = 2,
                      consistency_weight: float = 1.0,
                      ema_decay: float = 0.99,
                      consistency_ramp_up_epochs: int = 50,
                      save_interval: int = 1,
                      use_confidence_mask: bool = False,
                      confidence_threshold: float = 0.95,
                      use_entropy_filtering: bool = False,
                      entropy_threshold: float = 0.5,
                      num_gpus: int = 1,
                      continue_training: bool = False,
                      only_run_validation: bool = False,
                      disable_checkpointing: bool = False,
                      val_with_best: bool = False,
                      device: torch.device = torch.device('cuda')):
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                raise e

    if val_with_best:
        assert not disable_checkpointing

    if unlabeled_data_path is None:
        ds_name = maybe_convert_to_dataset_name(dataset_name_or_id if isinstance(dataset_name_or_id, str) else str(dataset_name_or_id))
        unlabeled_data_path = join(nnUNet_preprocessed, ds_name, 'SeminnUNet')

    semi_kwargs = {
        'unlabeled_data_path': unlabeled_data_path,
        'unlabeled_batch_size': unlabeled_batch_size,
        'consistency_weight': consistency_weight,
        'ema_decay': ema_decay,
        'consistency_ramp_up_epochs': consistency_ramp_up_epochs,
        'save_interval': save_interval,
        'use_confidence_mask': use_confidence_mask,
        'confidence_threshold': confidence_threshold,
        'use_entropy_filtering': use_entropy_filtering,
        'entropy_threshold': entropy_threshold,
    }

    if num_gpus > 1:
        assert device.type == 'cuda'
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            os.environ['MASTER_PORT'] = port

        mp.spawn(run_ddp,
                 args=(dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier,
                       disable_checkpointing, continue_training, only_run_validation, teacher_checkpoint,
                       num_gpus, semi_kwargs),
                 nprocs=num_gpus,
                 join=True)
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, device=device)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        _apply_semi_params(nnunet_trainer, **semi_kwargs)

        assert not (continue_training and only_run_validation)

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, teacher_checkpoint)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(False)


def run_semi_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str)
    parser.add_argument('configuration', type=str)
    parser.add_argument('fold', type=str)
    parser.add_argument('-tr', type=str, required=False, default='SemiSupervisedTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans')
    parser.add_argument('-teacher_checkpoint', type=str, required=False, default=None)
    parser.add_argument('-u', '--unlabeled_data_path', type=str, required=False, default=None)
    parser.add_argument('--unlabeled_batch_size', type=int, required=False, default=2)
    parser.add_argument('--ema_decay', type=float, required=False, default=0.99)
    parser.add_argument('--consistency_weight', type=float, required=False, default=1.0)
    parser.add_argument('--consistency_ramp_up_epochs', type=int, required=False, default=50)
    parser.add_argument('--save_interval', type=int, required=False, default=1)
    parser.add_argument('--use_confidence_mask', action='store_true', required=False)
    parser.add_argument('--confidence_threshold', type=float, required=False, default=0.95)
    parser.add_argument('--use_entropy_filtering', action='store_true', required=False)
    parser.add_argument('--entropy_threshold', type=float, required=False, default=0.5)
    parser.add_argument('-num_gpus', type=int, default=1, required=False)
    parser.add_argument('--c', action='store_true', required=False)
    parser.add_argument('--val', action='store_true', required=False)
    parser.add_argument('--val_best', action='store_true', required=False)
    parser.add_argument('--disable_checkpointing', action='store_true', required=False)
    parser.add_argument('-device', type=str, default='cuda', required=False)
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps']
    if args.device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_semi_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p,
                      args.teacher_checkpoint, args.unlabeled_data_path, args.unlabeled_batch_size,
                      args.consistency_weight, args.ema_decay, args.consistency_ramp_up_epochs, args.save_interval,
                      args.use_confidence_mask, args.confidence_threshold, args.use_entropy_filtering,
                      args.entropy_threshold, args.num_gpus, args.c, args.val, args.disable_checkpointing,
                      args.val_best, device=device)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    run_semi_training_entry()
