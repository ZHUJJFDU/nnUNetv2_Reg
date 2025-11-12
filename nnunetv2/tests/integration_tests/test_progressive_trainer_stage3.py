import inspect
import warnings
import numpy as np
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from nnunetv2.training.PKD.progressive_trainer import ProgressiveKDTrainer
from nnunetv2.regression.reg_dataloader import RegnnUNetDataLoader

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


class DummyRegnnUNetDataset:
    """
    轻量级虚拟数据集，用于测试 Stage3 的回归数据加载逻辑。
    提供 load_case(key) 接口，返回 (data, seg, properties, regression_value)。
    """

    def __init__(self, num_cases: int = 4, img_shape=(1, 64, 64, 64)):
        self.keys = [f"case_{i}" for i in range(num_cases)]
        self.img_shape = img_shape  # (C, Z, Y, X)

    def load_case(self, key: str):
        # data: (C, Z, Y, X)
        data = np.random.randn(*self.img_shape).astype(np.float32)
        # seg: (1, Z, Y, X) — 简单全零标签
        seg = np.zeros(self.img_shape[1:], dtype=np.uint8)[None, ...]
        properties = {
            "patient_id": key,
            "spacing": [1.0, 1.0, 1.0],
            "size_after_resampling": list(self.img_shape[1:]),
            "original_spacing": [1.0, 1.0, 1.0],
            "original_size": list(self.img_shape[1:]),
        }
        regression_value = float(np.random.rand())
        return data, seg, properties, regression_value

    def __len__(self):
        return len(self.keys)


def _build_trainer_for_stage3(num_threads: int):
    """使用 __new__ 构造不触发磁盘读取的 ProgressiveKDTrainer 并注入必要属性。"""
    t = ProgressiveKDTrainer.__new__(ProgressiveKDTrainer)
    # 必要的通用属性
    t.stage = 3
    t.transforms = None
    t.use_progress_bar = False
    # 数据集注入
    t.tr_dataset = DummyRegnnUNetDataset(num_cases=3)
    t.val_dataset = DummyRegnnUNetDataset(num_cases=2)
    # dataloader 所需参数
    t.patch_size = (32, 32, 32)
    t.final_patch_size = (32, 32, 32)
    t.label_manager = None
    t.batch_size = 2
    t.oversample_foreground_percent = 0.0
    t.memmap_mode = "r"
    t.num_threads_in_multithreaded = num_threads
    t.shuffle = True
    return t


def test_reg_dataloader_signature_no_transforms():
    """确保 RegnnUNetDataLoader 构造函数不包含 transforms 参数。"""
    sig = inspect.signature(RegnnUNetDataLoader.__init__)
    assert "transforms" not in sig.parameters, "RegnnUNetDataLoader 应不接受 'transforms' 参数"


def test_stage3_dataloaders_single_thread():
    """验证 num_threads=0 时选择 SingleThreadedAugmenter 并可取到首个批次。"""
    t = _build_trainer_for_stage3(num_threads=0)
    t.get_dataloaders()
    assert isinstance(t.dataloader_train, SingleThreadedAugmenter), "num_threads=0 应为 SingleThreadedAugmenter"
    # 取一个训练批次
    batch = next(t.dataloader_train)
    assert isinstance(batch, dict)
    assert "data" in batch and "target" in batch and "regression_value" in batch
    data = batch["data"]
    assert data.ndim == 5, f"数据维度应为 (B,C,Z,Y,X)，实际 {data.shape}"
    assert data.shape[1] == 1, "虚拟数据通道数应为 1"
    # 验证验证集也能预取首个批次
    val_batch = next(t.dataloader_val)
    assert "regression_value" in val_batch


def test_stage3_dataloaders_multi_thread():
    """验证 num_threads>0 时选择 MultiThreadedAugmenter 并可取到首个批次。"""
    t = _build_trainer_for_stage3(num_threads=2)
    t.get_dataloaders()
    assert isinstance(t.dataloader_train, MultiThreadedAugmenter), "num_threads>0 应为 MultiThreadedAugmenter"
    batch = next(t.dataloader_train)
    assert "regression_value" in batch


def run_all():
    warnings.filterwarnings("ignore")
    print("Running test_reg_dataloader_signature_no_transforms ...")
    test_reg_dataloader_signature_no_transforms()
    print("Running test_stage3_dataloaders_single_thread ...")
    test_stage3_dataloaders_single_thread()
    print("Running test_stage3_dataloaders_multi_thread ...")
    test_stage3_dataloaders_multi_thread()
    print("All ProgressiveKDTrainer Stage3 dataloader tests passed.")


if __name__ == "__main__":
    run_all()