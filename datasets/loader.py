import torch, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from datasets.build import build_dataset

def _loader_kwargs(cfg, split: str) -> dict:
    """Return the DataLoader arguments for a given split."""
    # ------------------------------------------------------------------
    # 1. Pull the split‑specific cfg values (fallback to TEST if unknown)
    # ------------------------------------------------------------------
    if split == "train":
        bs = cfg.TRAIN.BATCH_SIZE
        shuf = cfg.TRAIN.SHUFFLE
        drop = cfg.TRAIN.DROP_LAST
    elif split == "val":
        bs = cfg.VAL.BATCH_SIZE
        shuf = cfg.VAL.SHUFFLE
        drop = cfg.VAL.DROP_LAST
    elif split == "test":
        bs = cfg.TEST.BATCH_SIZE
        shuf = cfg.TEST.SHUFFLE
        drop = cfg.TEST.DROP_LAST
    else:
        raise ValueError(f"Unknown split: {split}")

    # ------------------------------------------------------------------
    # 2. Hard‑code the safe defaults for Colab T4
    # ------------------------------------------------------------------
    return dict(
        batch_size=bs,
        shuffle=shuf,
        drop_last=drop,
        num_workers=0,                     # <-- **CRITICAL** for /dev/shm
        pin_memory=getattr(cfg.DATA_LOADER, "PIN_MEMORY", False),
    )

def _load_csv_to_tensor(csv_path, dtype=torch.float32):
    df = pd.read_csv(csv_path, low_memory=False)
    tensor = torch.from_numpy(df.select_dtypes(include="number").values).to(dtype)
    return tensor, df.select_dtypes(include="number").columns.tolist()

def build_tensor_dataset(cfg, split):
    path = Path(cfg.DATA.PATH[split])
    tensor, cols = _load_csv_to_tensor(path)
    cfg.DATA.FEATURES = cols
    return TensorDataset(tensor)

def construct_loader(cfg, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    bs   = getattr(cfg, split.upper()).BATCH_SIZE
    shuf = getattr(cfg, split.upper()).SHUFFLE
    drop = getattr(cfg, split.upper()).DROP_LAST
    dataset = build_tensor_dataset(cfg, split)
    return DataLoader(dataset,
                      batch_size=bs,
                      shuffle=shuf,
                      drop_last=drop,
                      num_workers=0,
                      pin_memory=False)

def get_train_dataloader(cfg):
    return construct_loader(cfg, "train")


def get_val_dataloader(cfg):
    return construct_loader(cfg, "val")


def get_test_dataloader(cfg):
    return construct_loader(cfg, "test")
