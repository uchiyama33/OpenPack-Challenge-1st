#%%
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict,
    eval_operation_segmentation_wrapper,
    make_submission_zipfile,
)
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import splits_5_fold_new
from utils.datamodule import OpenPackAllDataModule
from utils.lightning_module import TransformerPL

#%%


_ = optk.utils.notebook.setup_root_logger()
logger = logging.getLogger(__name__)

logger.debug("debug")
logger.info("info")
logger.warning("warning")
optorch.configs.register_configs()

issue = "transformer_plusLSTM-d6-h16-np150-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu400_keypoint500-ht12_printer12-"
config_dir = os.path.join("/workspace/logs/all/transformer_plusLSTM/", issue, ".hydra")
with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = hydra.compose(
        config_name="config.yaml",
        # config_name="unet-tutorial2.yaml",
    )
cfg.dataset.annotation.activity_sets = dict()  # Remove this attribute just for the simpler visualization.
cfg.dataset.split = optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT  # DEBUG_SPLIT
# cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
optorch.utils.reset_seed(seed=0)

#%%
# class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
#     dataset_class = optorch.data.datasets.OpenPackImu

#     def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
#         kwargs = {
#             "window": self.cfg.train.window,
#             "debug": self.cfg.debug,
#         }
#         return kwargs


# datamodule = OpenPackImuDataModule(cfg)
# datamodule.setup("test")
# dataloaders = datamodule.test_dataloader()

# batch = dataloaders[0].dataset.__getitem__(0)

#%%
device = torch.device("cuda")
logdir = Path(cfg.path.logdir.rootdir)
logger.debug(f"logdir = {logdir}")

num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs
# num_epoch = 20 # NOTE: Set epochs manually for debugging

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=num_epoch,
    logger=False,  # disable logging module
    default_root_dir=logdir,
    enable_progress_bar=False,  # disable progress bar
    enable_checkpointing=True,
)
logger.debug(f"logdir = {logdir}")

#%%
def delete_overlap(y, te, unixtime):  # TODO 1000以外のtime_step_width対応
    b, c, t = y.shape
    unixtime = unixtime.ravel()
    te = te.ravel()
    y = y.transpose(0, 2, 1).reshape(b * t, c)
    delta = unixtime[1:] - unixtime[:-1]
    idx_list = np.where(delta != 1000)[0]
    n_del = 0
    while len(idx_list) > 0:
        idx = idx_list[0]
        y = np.delete(y, slice(idx, idx + 2), 0)
        unixtime = np.delete(unixtime, slice(idx, idx + 2), 0)
        te = np.delete(te, slice(idx, idx + 2), 0)

        delta = unixtime[1:] - unixtime[:-1]
        idx_list = np.where(delta != 1000)[0]
        n_del += 1

    unixtime = np.concatenate([unixtime, np.repeat(unixtime[-1], n_del * 2)])
    unixtime = unixtime.reshape(b, t)
    te = np.concatenate([te, np.repeat(te[-1], n_del * 2)])
    te = te.reshape(b, t)
    y = np.concatenate([y, np.repeat([y[-1]], n_del * 2, 0)])
    y = y.reshape(b, t, c).transpose(0, 2, 1)
    return y, te, unixtime


def average_slide_results(slide_results):
    y_list = [r["y"] for r in slide_results]
    t_list = [r["t"] for r in slide_results]
    unixtime_list = [r["unixtime"] for r in slide_results]

    B, C, T = y_list[0].shape

    unixtime_for_use = unixtime_list[0].ravel()
    y_stack = [[] for i in range(len(unixtime_for_use))]
    t_stack = [[] for i in range(len(unixtime_for_use))]

    unixtime_for_use = unixtime_list[0].ravel()

    for y, t, unixtime in zip(y_list, t_list, unixtime_list):
        y = y.transpose(0, 2, 1)
        for _y, _t, _unixtime in zip(y, t, unixtime):
            for __y, __t, __unixtime in zip(_y, _t, _unixtime):
                if __unixtime in unixtime_for_use:
                    ind = (unixtime_for_use == __unixtime).argmax()
                    y_stack[ind].append(__y)
                    t_stack[ind].append(__t)

    y_mean = [None] * len(unixtime_for_use)
    for i in range(len(unixtime_for_use)):
        y_mean[i] = softmax(np.stack(y_stack[i]), 1).mean(0)
    y_mean = np.array(y_mean).reshape(B, T, C).transpose(0, 2, 1)

    return y_mean


# %%　パッチずらしあり

results = []
nums_folds = 5

# for k in range(nums_folds):
for k in range(1):
    cfg.dataset.split = getattr(splits_5_fold_new, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
    datamodule = OpenPackAllDataModule(cfg)
    datamodule.set_fold(k)
    cfg.mode = "test"
    datamodule.setup("test")
    dataloaders = datamodule.test_dataloader()
    split = cfg.dataset.split.test

    outputs = dict()
    chk_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}", "*")
    chk_path = glob.glob(chk_dir)[0]
    plmodel = TransformerPL.load_from_checkpoint(chk_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)
    plmodel.eval()
    plmodel.set_fold(k)
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        slide_results = []
        for n in tqdm(range(cfg.model.num_patches)):
            dataloader.dataset.set_test_start_time(n)

            with torch.inference_mode():
                trainer.test(plmodel, dataloader)

            # save model outputs
            pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
            pred_dir.mkdir(parents=True, exist_ok=True)

            for key, arr in plmodel.test_results.items():
                path = Path(pred_dir, f"{key}.npy")
                np.save(path, arr)
                logger.info(f"save {key}[shape={arr.shape}] to {path}")

            t = plmodel.test_results.get("t")
            y = plmodel.test_results.get("y")
            unixtime = plmodel.test_results.get("unixtime")

            y, t, unixtime = delete_overlap(y, t, unixtime)
            slide_results.append({"y": y, "t": t, "unixtime": unixtime})

        y = average_slide_results(slide_results)
        t = slide_results[0]["t"]
        unixtime = slide_results[0]["unixtime"]

        outputs[f"{user}-{session}"] = {
            "t": t,
            "y": y,
            "unixtime": unixtime,
        }
        results.append(outputs)

# %%
df_summary = eval_operation_segmentation_wrapper(
    cfg,
    outputs,
    OPENPACK_OPERATIONS,
)
# %%
df_summary[df_summary["key"] == "all"]

# %%    パッチずらしなし

results = []
nums_folds = 5

# for k in range(nums_folds):
for k in range(1):
    cfg.dataset.split = getattr(splits_5_fold_new, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
    datamodule = OpenPackAllDataModule(cfg)
    datamodule.set_fold(k)
    cfg.mode = "test"
    datamodule.setup("test")
    dataloaders = datamodule.test_dataloader()
    split = cfg.dataset.split.test

    outputs = dict()
    chk_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}", "*")
    chk_path = glob.glob(chk_dir)[0]
    plmodel = TransformerPL.load_from_checkpoint(chk_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)
    plmodel.eval()
    plmodel.set_fold(k)
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        dataloader.dataset.set_test_start_time(0)

        with torch.inference_mode():
            trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            path = Path(pred_dir, f"{key}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        t = plmodel.test_results.get("t")
        y = plmodel.test_results.get("y")
        unixtime = plmodel.test_results.get("unixtime")

        y, t, unixtime = delete_overlap(y, t, unixtime)

        outputs[f"{user}-{session}"] = {
            "t": t,
            "y": y,
            "unixtime": unixtime,
        }
        results.append(outputs)

# %%
df_summary = eval_operation_segmentation_wrapper(
    cfg,
    outputs,
    OPENPACK_OPERATIONS,
)
# %%
df_summary[df_summary["key"] == "all"]
# %%
def plot_timeline(samples, t_idx, y_softmax, title_prefix=""):
    fig, ax0 = plt.subplots(1, 1, figsize=(20, 4))

    prob = y_softmax[slice(*samples)].transpose(1, 0, 2).reshape(11, -1)
    gt = t_idx[slice(*samples)].ravel()
    pred = prob.argmax(axis=0)

    print(f"prob={prob.shape} pred={pred.shape}, gt={gt.shape}")
    seq_len = prob.shape[1]

    # -- Prob --
    sns.heatmap(prob, vmin=0, vmax=1.0, cmap="viridis", cbar=False, ax=ax0)

    # -- Ground Truth --
    x = np.arange(seq_len)
    ax0.plot(
        x,
        gt + 0.5,
        label="ground truth",
        linewidth=5,
        color="C3",
    )
    # ax0.plot(
    #     x, pred + 0.5, label="pred",
    #     linewidth=1, color="C1", alpha=0.5,
    # )

    # -- Style --
    ax0.invert_yaxis()
    xticks = np.arange(0, seq_len, 30 * 30)
    xticks_minor = np.arange(0, seq_len, 30 * 10)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticks // 30 + samples[0] * 60, rotation=0)
    ax0.set_xticks(xticks_minor, minor=True)

    ax0.set_yticks(np.arange(11) + 0.5)
    ax0.set_yticklabels(np.arange(11), rotation=0)

    ax0.set_xlabel("Time [s]", fontsize="large", fontweight="bold")
    ax0.set_ylabel("Class Index", fontsize="large", fontweight="bold")
    ax0.set_title(
        f"{title_prefix} | {samples[0]}min ~ {samples[1]}min", fontsize="xx-large", fontweight="bold"
    )
    ax0.grid(True, which="minor", linestyle=":")
    ax0.legend(loc="upper right")

    fig.tight_layout()
    return fig


# %%
key = "U0102-S0500"

unixtimes = outputs[key]["unixtime"]
t_idx = outputs[key]["t"]
y = outputs[key]["y"]
y_softmax = softmax(y, axis=1)
print(f"unixtimes={unixtimes.shape}, t_idx={t_idx.shape}, y={y.shape}, y_softmax={y_softmax.shape}")

# %%
samples = (35, 36)

fig = plot_timeline(samples, t_idx, y_softmax, title_prefix=f"{cfg.model.name} | {user} {session}")
fig.show()

# %%
