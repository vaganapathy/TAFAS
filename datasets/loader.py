from torch.utils.data import DataLoader

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

def construct_loader(cfg, split):
    dataset = build_dataset(cfg, split)
    # if split == "train":
    #     batch_size = cfg.TRAIN.BATCH_SIZE
    #     shuffle = cfg.TRAIN.SHUFFLE
    #     drop_last = cfg.TRAIN.DROP_LAST
    # elif split == "val":
    #     batch_size = cfg.VAL.BATCH_SIZE
    #     shuffle = cfg.VAL.SHUFFLE
    #     drop_last = cfg.VAL.DROP_LAST
    # elif split == "test":
    #     batch_size = cfg.TEST.BATCH_SIZE
    #     shuffle = cfg.TEST.SHUFFLE
    #     drop_last = cfg.TEST.DROP_LAST
    # else:
    #     raise ValueError


    loader = DataLoader(dataset, **_loader_kwargs(cfg, split))  
    return loader


def get_train_dataloader(cfg):
    return construct_loader(cfg, "train")


def get_val_dataloader(cfg):
    return construct_loader(cfg, "val")


def get_test_dataloader(cfg):
    return construct_loader(cfg, "test")
