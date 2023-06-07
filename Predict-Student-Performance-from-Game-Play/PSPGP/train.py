import gc
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.optim import lr_scheduler

from config import *
from utils import build_dataloaders, Evaluator, MainTrainer
from model import EventConvSimple
import warnings
warnings.simplefilter("default")


def main():
    cfg = CFG()
    X = pd.read_parquet('../data/train_part.parquet')
    y_lvgp = pd.read_parquet('../data/labels_lvgp.parquet')
    # y_all = pd.read_parquet('./data/labels_all.parquet')
    if cfg.train:
        sess_id = X["session_id"].unique()

        oof_pred = pd.DataFrame(np.zeros((len(sess_id), N_QNS)), index=sess_id)
        cv = GroupKFold(n_splits=5)
        for i, (tr_idx, val_idx) in enumerate(cv.split(X=X, groups=X["session_id"])):
            print(f"Training and evaluation process of fold{i} starts...")

            # Prepare data
            X_tr, X_val = X.iloc[tr_idx, :], X.iloc[val_idx, :]
            sess_tr, sess_val = X_tr["session_id"].unique(), X_val["session_id"].unique()
            y_tr, y_val = y_lvgp[y_lvgp["session_id"].isin(sess_tr)], y_lvgp[y_lvgp["session_id"].isin(sess_val)]

            # Run level_group-wise modeling
            oof_pred_fold = []
            for lv_gp in LEVEL_GROUP:
                print(f"=====LEVEL GROUP {lv_gp}=====")
                qn_idx = QNS_PER_LV_GP[lv_gp]  # Question index
                lvs = LV_PER_LV_GP[lv_gp]
                X_tr_, X_val_ = X_tr[X_tr["level_group"] == lv_gp], X_val[X_val["level_group"] == lv_gp]
                y_tr_, y_val_ = y_tr[y_tr["level_group"] == lv_gp], y_val[y_val["level_group"] == lv_gp]

                # Build dataloader
                train_loader, val_loader = build_dataloaders((X_tr_, y_tr_), (X_val_, y_val_), cfg.BATCH_SIZE, **{"t_window": cfg.T_WINDOW})

                # Build model
                model = EventConvSimple(len(lvs), len(qn_idx), **{"cat_feats": cfg.CAT_FEATS})
                model.to(cfg.DEVICE)

                # Build criterion
                loss_fn = nn.BCEWithLogitsLoss()

                # Build solvers
                optimizer = optim.Adam(list(model.parameters()), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
                lr_skd = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, eta_min=1e-5, T_mult=1)

                # Build evaluator
                evaluator = Evaluator(cfg.EVAL_METRICS, len(qn_idx))

                # Build trainer
                trainer_cfg = {
                    "cfg": cfg,
                    "model": model,
                    "loss_fn": loss_fn,
                    "optimizer": optimizer,
                    "lr_skd": lr_skd,
                    "evaluator": evaluator,
                    "train_loader": train_loader,
                    "eval_loader": val_loader,
                }
                trainer = MainTrainer(**trainer_cfg)

                # Run training and evaluation processes for one fold
                best_model, best_preds = trainer.train_eval(lv_gp)
                oof_pred_fold.append(best_preds["val"])

                # Dump output objects of the current fold
                torch.save(best_model.state_dict(), f"fold{i}_{lv_gp}")

                # Free mem.
                del (X_tr_, X_val_, y_tr_, y_val_, train_loader, val_loader,
                     model, loss_fn, optimizer, lr_skd, evaluator, trainer)
                _ = gc.collect()

            # Record oof prediction of the current fold
            oof_pred.loc[sess_val, :] = torch.cat(oof_pred_fold, dim=1).numpy()
    else:
        oof_pred = pd.read_csv("../data/preds.csv")
        oof_pred.set_index("session", inplace=True)
        oof_pred.rename({"session": "session_id"}, axis=1, inplace=True)


if __name__ == '__main__':
    main()
