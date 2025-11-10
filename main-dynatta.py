import os

from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from utils.log import init_wandb
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
# from tta.tafas import build_adapter
from tta.DynaTTA import build_adapter
from config import get_norm_module_cfg
from utils.misc import mkdir
import csv

def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # set wandb logger
    if cfg.WANDB.ENABLE:
        init_wandb(cfg)

    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    adapt_test_mse, adapt_test_mae, results = None, None, None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()
        
    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    if cfg.TTA.ENABLE:
        adapter = build_adapter(cfg, model, norm_module=norm_module)
        adapt_test_mse, adapt_test_mae = adapter.adapt()
    if cfg.TEST.ENABLE:
        predictor = Predictor(cfg, model, norm_module=norm_module)
        results = predictor.predict()

    if(adapt_test_mse and adapt_test_mae and results):
        csv_file_name = os.path.join(mkdir(cfg.CSV_DIR) / f"{cfg.MODEL.NAME}.csv")
        if not os.path.isfile(csv_file_name):
            with open(csv_file_name, "a", newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(["model", "data", "seq_len", "pred_len", "test_mse", "test_mae", "train_mse", "train_mae", "adapt_test_mse", "adapt_test_mae", "imp mse", "imp mae"])
        imp_mse = ((adapt_test_mse - results["test_mse"])/results["test_mse"] * -1) * 100
        imp_mae = ((adapt_test_mae - results["test_mae"])/results["test_mae"] * -1) * 100
        with open(csv_file_name, "a", newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([cfg.MODEL.NAME, cfg.DATA.NAME, cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN] + [results["test_mse"], results["test_mae"], results["train_mse"], results["train_mae"]] + [adapt_test_mse, adapt_test_mae, imp_mse, imp_mae])



if __name__ == '__main__':
    main()