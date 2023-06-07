import gc
import os
import pickle
import sys
from abc import abstractmethod
from copy import deepcopy
from random import random

from sklearn.metrics import f1_score
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torchmetrics import AUROC
from config import *

import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List, Tuple, Type, Union, Callable


cfg = CFG()


def seed_all(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Parameters:
        seed: manually specified seed number

    Return:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def summarize(
        df: pd.DataFrame,
        file_name: Optional[str] = None,
        n_rows_to_display: Optional[int] = 5,
) -> None:
    """Summarize DataFrame.

    Parameters:
        df: input data
        file_name: name of the input file
        n_rows_to_display: number of rows to display

    Return:
        None
    """
    file_name = "Data" if file_name is None else file_name

    # Derive NaN ratio for each column
    nan_ratio = pd.isna(df).sum() / len(df) * 100
    nan_ratio.sort_values(ascending=False, inplace=True)
    nan_ratio = nan_ratio.to_frame(name="NaN Ratio").T

    # Derive zero ratio for each column
    zero_ratio = (df == 0).sum() / len(df) * 100
    zero_ratio.sort_values(ascending=False, inplace=True)
    zero_ratio = zero_ratio.to_frame(name="Zero Ratio").T

    # Print out summarized information
    print(f"=====Summary of {file_name}=====")
    print(df.head(n_rows_to_display))
    print(f"Shape: {df.shape}")
    print("NaN ratio:")
    print(nan_ratio)
    print("Zero ratio:")
    print(zero_ratio)


def drop_multi_game_naive(
        df: pd.DataFrame,
        local: bool = True
) -> pd.DataFrame:
    """Drop events not occurring at the first game play.

    Note: `groupby` should be done with `sort=False` for the new
    training set, because `session_id` isn't sorted by default.

    Parameters:
        df: input DataFrame
        local: if False, the processing step is simplified based on the
            properties of returned test DataFrame by time series API

    Return:
        df: DataFrame with events occurring at the first game play only
    """
    df = df.copy()
    if local:
        df["lv_diff"] = df.groupby("session_id", sort=False).apply(lambda x: x["level"].diff().fillna(0)).values
    else:
        df["lv_diff"] = df["level"].diff().fillna(0)
    reversed_lv_pts = df["lv_diff"] < 0
    df.loc[~reversed_lv_pts, "lv_diff"] = 0
    if local:
        df["multi_game_flag"] = df.groupby("session_id", sort=False)["lv_diff"].cumsum().values
    else:
        df["multi_game_flag"] = df["lv_diff"].cumsum()
    multi_game_mask = df["multi_game_flag"] < 0
    multi_game_rows = df[multi_game_mask].index
    df = df.drop(multi_game_rows).reset_index(drop=True)

    # Drop redundant columns
    df.drop(["lv_diff", "multi_game_flag"], axis=1, inplace=True)

    return df


def map_lvgp_order(lvgp_seq: pd.Series) -> pd.Series:
    """Map level_group sequence to level_group order sequence.

    Parameters:
        lvgp_seq: level_group sequence

    Return:
        lvgp_order_seq: level_group order sequence
    """
    lvgp_order_seq = lvgp_seq.map(LVGP_ORDER)

    return lvgp_order_seq


def check_multi_game(df: pd.DataFrame) -> bool:
    """Check if multiple game plays exist in any session.

    Parameters:
        df: input DataFrame

    Return:
        multi_game_exist: if True, multiple game plays exist in at
            least one of the session
    """
    multi_game_exist = False
    for i, (sess_id, gp) in enumerate(df.groupby("session_id", sort=False)):
        if ((not gp["level"].is_monotonic_increasing)
                or (not gp["lvgp_order"].is_monotonic_increasing)):
            multi_game_exist = True
            break

    return multi_game_exist


def get_factorize_map(
    values: Union[np.ndarray, pd.Series, pd.Index],
    sort: bool = True
) -> Dict[str, int]:
    """Factorize array and return numeric representation map.

    Parameters:
        values: 1-D sequence to factorize
        sort: whether to sort unique values

    Return:
        val2code: mapping from value to numeric code
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    vals_uniq = values.unique()
    if sort:
        vals_uniq = sorted(vals_uniq)
    val2code = {val: code for code, val in enumerate(vals_uniq)}

    return val2code


def feature_engineer(
    df: pd.DataFrame,
    train: bool,
    cfg
) -> pd.DataFrame:

    df["et_diff"] = df.groupby("session_id", sort=False).apply(lambda x: x["elapsed_time"].diff().fillna(0)).values
    df["et_diff"] = df["et_diff"].clip(0, 3.6e6)

    # Factorize categorical features
    df["event_comb"] = df["event_name"] + "_" + df["name"]
    for cat_feat in cfg.CAT_FEATS:
        orig_col = cat_feat[:-5]
        if train:
            cat2code = get_factorize_map(df[orig_col])
            with open(f"./cat2code/{orig_col}2code.pkl", "wb") as f:
                pickle.dump(cat2code, f)
        else:
            with open(f"./cat2code/{orig_col}2code.pkl", "rb") as f:
                cat2code = pickle.load(f)
        df[cat_feat] = df[orig_col].map(cat2code)
    return df[["session_id", "level_group", "level"] + cfg.FEATS]


class LvGpDataset(Dataset):
    """Dataset for `level_group`-wise modeling.

    Input data are all from the same `level_group`. Also, if `t_window`
    isn't long enough, only raw features from the last level of a
    `level_group` is retrieved.

    Commonly used notations are summarized as follows:
    * `N` denotes the number of samples
    * `P` denotes the length of lookback time window
    * `Q` denotes the number of questions in the current level group
    * `C` denotes the number of channels (numeric features)
        *Note: Currently, only `et_diff` is used as the numeric feature
    * `M` denotes the number of categorical features

    Each sample is session-specific, which is illustrated as follows:

    Let C=1 (i.e., only `et_diff` is used as the feature) and P=60, we
    have data appearance like:

    idx  t-59  t-58  ...  t-1  t
    0     1     2          0   2   -> session_id == 20090312431273200
    1     2     3          4   1   -> session_id == 20090312433251036
    .
    .
    .
    N-2
    N-1

    Parameters:
        data: input data
        t_window: length of lookback time window
    """

    n_samples: int

    def __init__(
            self,
            data: Tuple[pd.DataFrame, pd.DataFrame],
            t_window: int,
            **kwargs: Any,
    ) -> None:
        self.X_base, self.y_base = data
        #         self.y_base["correct"] = self.y_base["correct"].apply(lambda x: ast.literal_eval(x))
        self.t_window = t_window

        # Specify features to use
        self.feats = [feat for feat in cfg.FEATS if feat not in cfg.CAT_FEATS]
        self.cat_feats = cfg.CAT_FEATS

        # Setup level-related metadata
        self.lv_gp = self.X_base["level_group"].unique()[0]
        self.levels = LV_PER_LV_GP[self.lv_gp]
        self.n_lvs = len(self.levels)

        # Generate data samples
        self._chunk_X_y()
        self._set_n_samples()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        X = self.X[idx, ...]
        y = self.y[idx, ...]
        X_cat = self.X_cat[idx, ...]
        stamp = self.stamp

        data_sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "X_cat": torch.tensor(X_cat, dtype=torch.int32),
            "y": torch.tensor(y, dtype=torch.float32),
            "stamp": torch.tensor(stamp, dtype=torch.int32)
        }

        return data_sample

    def _set_n_samples(self) -> None:
        """Derive the number of samples."""
        self.n_samples = len(self.X)

    def _chunk_X_y(self) -> None:
        """Chunk data samples."""
        X, y = [], []
        X_cat = []
        stamp = []

        for sess_id in self.y_base["session_id"].values:
            # Target columns hard-coded temporarily
            X_sess = self.X_base[self.X_base["session_id"] == sess_id]

            pad_len = 0
            x_num = []
            for i, feat in enumerate(self.feats):
                x_num_i = X_sess[feat].values[-self.t_window:]
                if i == 0 and len(x_num_i) < self.t_window:
                    pad_len = self.t_window - len(x_num_i)

                if pad_len != 0:
                    x_num_i = np.pad(x_num_i, (pad_len, 0), "constant")

                x_num.append(x_num_i)

            x_cat = X_sess[self.cat_feats].values[-self.t_window:]
            if pad_len != 0:
                x_cat = np.pad(x_cat, ((pad_len, 0), (0, 0)), "constant", constant_values=-1)  # (P, M)

            for st, end in [(0, 2), (2, 4), (4, 6), (6, 8)]:
                stamp.append(int(str(sess_id)[st:end]))

            x_num = np.stack(x_num, axis=1)  # (P, C)
            X.append(x_num)
            X_cat.append(x_cat)
            y.append(self.y_base[self.y_base["session_id"] == sess_id]["correct"].values[0])

        self.X = np.stack(X)  # (N, P, C)
        self.X_cat = np.stack(X_cat)  # (N, P, M)
        self.y = np.vstack(y)  # (N, Q)
        self.stamp = stamp  # (N, D)


def build_dataloaders(
    data_tr: Tuple[pd.DataFrame, pd.DataFrame],
    data_val: Tuple[pd.DataFrame, pd.DataFrame],
    batch_size: int,
    **dataset_cfg: Any,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation dataloaders.

    Parameters:
        data_tr: training data
        data_val: validation data
        dataloader_cfg: hyperparameters of dataloader
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training dataloader
        val_loader: validation dataloader
    """
    if data_tr is not None:
        train_loader = DataLoader(
            LvGpDataset(data_tr, **dataset_cfg),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=None,
        )
    else:
        train_loader = None
    if data_val is not None:
        val_loader = DataLoader(
            LvGpDataset(data_val, **dataset_cfg),
            batch_size=batch_size,
            shuffle=False,  # Hard-coded
            collate_fn=None,
        )
    else:
        val_loader = None

    return train_loader, val_loader


class Evaluator(object):
    """Evaluator.

    Parameters:
        metric_names: evaluation metrics
        n_qns: number of questions
    """

    eval_metrics: Dict[str, Callable[..., Union[float]]]

    def __init__(self, metric_names: List[str], n_qns: int):
        self.metric_names = metric_names
        self.n_qns = n_qns
        self.eval_metrics = {}
        self._build()

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Parameters:
            y_true: groundtruths
            y_pred: predicting results

        Return:
            eval_result: evaluation performance report
        """
        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            if metric_name == "f1":
                for thres in [0.63]:
                    eval_result[f"{metric_name}@{thres}"] = metric(y_pred, y_true, thres)
            else:
                eval_result[metric_name] = metric(y_pred, y_true)

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "auroc":
                self.eval_metrics[metric_name] = self._AUROC
            elif metric_name == "f1":
                self.eval_metrics[metric_name] = self._F1

    def _AUROC(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Area Under the Receiver Operating Characteristic curve.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths

        Return:
            auroc: area under the receiver operating characteristic
                curve
        """
        metric = AUROC(task="multilabel", num_labels=self.n_qns)
        _ = metric(y_pred, y_true.int())
        auroc = metric.compute().item()
        metric.reset()

        return auroc

    def _F1(self, y_pred: Tensor, y_true: Tensor, thres: float) -> float:
        """F1 score.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths
            thres: threshold to convert probability to bool

        Return:
            f1: F1 score
        """
        y_pred = (y_pred.numpy().reshape(-1, ) > thres).astype("int")
        y_true = y_true.numpy().reshape(-1, )
        f1 = f1_score(y_true, y_pred, average="macro")

        return f1


class BaseTrainer:
    """Base class for all customized trainers.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        evaluator: task-specific evaluator
    """

    def __init__(
        self,
        cfg: Type[CFG],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: _LRScheduler,
        evaluator: Evaluator,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.evaluator = evaluator

        self.device = cfg.DEVICE
        self.epochs = cfg.EPOCH

        # Model checkpoint
        self.ckpt_metric = cfg.CKPT_METRIC

        self._iter = 0
        self._track_best_model = True

    def train_eval(self, proc_id: Union[str, int]) -> Tuple[Module, Dict[str, Tensor]]:
        """Run training and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Return:
            best_model: model instance with the best monitored
                objective (e.g., the lowest validation loss)
            y_preds: inference results on different datasets
        """
        best_val_loss = 1e18
        best_epoch = 0
        try:
            best_model = deepcopy(self.model)
        except RuntimeError as e:
            best_model = None
            self._track_best_model = False
            print("In-memoey best model tracking is disabled.")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            val_metric_msg = ""
            for metric, score in val_result.items():
                val_metric_msg += f"{metric.upper()} {round(score, 4)} | "
            print(f"Epoch{epoch} | Training loss {train_loss:.4f} | Validation loss {val_loss:.4f} | {val_metric_msg}")

            # Record the best checkpoint
            ckpt_metric_val = val_result[self.ckpt_metric]
            ckpt_metric_val = -ckpt_metric_val
            if ckpt_metric_val < best_val_loss:
                print(f"Validation performance improves at epoch {epoch}!!")
                best_val_loss = ckpt_metric_val
                if self._track_best_model:
                    best_model = deepcopy(self.model)
                else:
                    self._save_ckpt()
                best_epoch = epoch

        # Run final evaluation
        if not self._track_best_model:
            self._load_best_ckpt()
            best_model = self.model
        else:
            self.model = best_model
        final_prf_report, y_preds = self._eval_with_best()
        self._log_best_prf(final_prf_report)

        return best_model, y_preds

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
                *Note: If multitask is used, returned object will be
                    a dictionary containing losses of subtasks and the
                    total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(
        self,
        return_output: bool = False,
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: if evaluation is run on test set, set it to True
                *Note: The setting is mainly used to disable DAE doping
                    during test phase.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        raise NotImplementedError

    def _eval_with_best(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tensor]]:
        """Run final evaluation process with the best checkpoint.

        Return:
            final_prf_report: performance report of final evaluation
            y_preds: inference results on different datasets
        """
        final_prf_report = {}
        y_preds = {}

        self._disable_shuffle()
        dataloaders = {"train": self.train_loader}
        if self.eval_loader is not None:
            dataloaders["val"] = self.eval_loader

        for datatype, dataloader in dataloaders.items():
            self.eval_loader = dataloader
            eval_loss, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[datatype] = eval_result
            y_preds[datatype] = y_pred

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _save_ckpt(self, proc_id: int = 0, save_best_only: bool = True) -> None:
        """Save checkpoints.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed
            save_best_only: only checkpoint of the best epoch is saved

        Return:
            None
        """
        torch.save(self.model.state_dict(), "model_tmp.pt")

    def _load_best_ckpt(self, proc_id: int = 0) -> None:
        """Load the best model checkpoint for final evaluation.

        The best checkpoint is loaded and assigned to `self.model`.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed

        Return:
            None
        """
        device = torch.device(self.device)
        self.model.load_state_dict(
            torch.load("model_tmp.pt", map_location=device)
        )
        self.model = self.model.to(device)

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Parameters:
            prf_report: performance report

        Return:
            None
        """
        import json

        print(">>>>> Performance Report - Best Ckpt <<<<<")
        print(json.dumps(prf_report, indent=4))


class MainTrainer(BaseTrainer):
    """Main trainer.

    Parameters:
        cfg: experiment configuration
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        train_loader: training data loader
        eval_loader: validation data loader
    """

    def __init__(
        self,
        cfg: Type[CFG],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: _LRScheduler,
        evaluator: Evaluator,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ):
        super(MainTrainer, self).__init__(
            cfg, model, loss_fn, optimizer, lr_skd, evaluator
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader, file=sys.stdout)):
            self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            x_cat = batch_data["X_cat"].to(self.device)
            y = batch_data["y"].to(self.device)
            stamp = batch_data["stamp"].to(self.device)

            # Forward pass
            output = self.model(x, x_cat, stamp)
            self._iter += 1

            # Derive loss
            loss = self.loss_fn(output, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            train_loss_total += loss.item()

            # Free mem.
            del x, y, output
            _ = gc.collect()

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            test: always ignored, exists for compatibility

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        eval_loss_total = 0
        y_true = None
        y_pred = None

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            x_cat = batch_data["X_cat"].to(self.device)
            y = batch_data["y"].to(self.device)
            stamp = batch_data["stamp"].to(self.device)

            # Forward pass
            output = self.model(x, x_cat, stamp)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            if i == 0:
                y_true = y.detach().cpu()
                y_pred = output.detach().cpu()
            else:
                # Avoid situ ation that batch_size is just equal to 1
                y_true = torch.cat((y_true, y.detach().cpu()))
                y_pred = torch.cat((y_pred, output.detach().cpu()))

            del x, y, output
            _ = gc.collect()

        y_pred = torch.sigmoid(y_pred)  # Tmp. workaround (because loss has built-in sigmoid)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)
        eval_result = self.evaluator.evaluate(y_true, y_pred)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None

    def derive_cv_score(
            y_true: pd.DataFrame,
            y_pred: pd.DataFrame,
            return_traces: bool = False
    ) -> Tuple[float, float, Optional[Tuple[List[float], List[float]]]]:
        """Compute the fianl CV score (i.e., f1 score).

        Parameters:
            y_true: groundtruths
            y_pred: predicting results
            return_traces: if True, f1 scores at different thresholds
                are returned

        Return:
            best_f1: final CV score at the best prob threshold
            best_thres: threshold with the best CV score
            f1s: f1 scores at different thresholds
            thresholds: thresholds to convert probability to bool
        """
        y_pred = y_pred.sort_index()
        thres_range = np.arange(0.2, 0.81, 0.01)

        f1s, thresholds = [], []
        best_f1 = 0.0
        best_thres = 0.0
        for thres in thres_range:
            y_pred_bool = (y_pred.values.reshape(-1, ) > thres).astype("int")
            f1 = f1_score(y_true.values.reshape(-1, ), y_pred_bool, average="macro")
            f1s.append(f1)
            thresholds.append(thres)

            if f1 > best_f1:
                best_f1 = f1
                best_thres = thres

        traces = (f1s, thresholds)

        if return_traces:
            return best_f1, best_thres, traces
        else:
            return best_f1, best_thres, None


