"""
Scripts to train CCS and LR probes.

Author: Artyom Karpov, www.artkpv.net
"""

from C3S import C3S
from CCS import CCS
from LogisticRegression import LogisticRegression
import logging
from truthful_qa_ds import create_dataloaders, get_hidden_state_datasets
from torch.utils.data import random_split, DataLoader

import lightning as pl
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, open_dict
from model import create_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Training with config: "{cfg}"')
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.probes_dir).mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg._device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(cfg._device)

    one_hs_ds, disj_hs_ds, conj_hs_ds = get_hidden_state_datasets(
        cfg, create_model=create_model
    )

    with open_dict(cfg):
        cfg._input_dim = one_hs_ds.tensors[0].shape[-1]

    if 'ccs' in cfg.probes:
        train_on_all_ccs(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds)
        train_ccs(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds)

    if 'lr' in cfg.probes:
        train_lr(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds)

    if 'c3s' in cfg.probes:
        train_c3s(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds)


def train_on_all_ccs(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds):
    # Merge three Tensor datasets into one:
    ds = torch.utils.data.ConcatDataset([one_hs_ds, disj_hs_ds, conj_hs_ds])

    probe = CCS(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.ccs.epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    train_dl, _ = create_dataloaders([ds], batch_size=cfg.ccs.batch_size)
    trainer.fit(probe, train_dl)

    torch.save(
        probe.state_dict(),
        f"{cfg.probes_dir}/truthful_qa_all_ccs_probe.pt",
    )


def train_ccs(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds):
    for ds, name in [
        (one_hs_ds, "one"),
        (disj_hs_ds, "disj"),
        (conj_hs_ds, "conj"),
    ]:
        train_dl, _ = create_dataloaders([ds], batch_size=cfg.ccs.batch_size)
        probe = CCS(cfg)
        trainer = pl.Trainer(
            max_epochs=cfg.ccs.epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(probe, train_dl)

        torch.save(
            probe.state_dict(),
            f"{cfg.probes_dir}/truthful_qa_{name}_ccs_probe.pt",
        )


def train_c3s(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds):
    # Merge one_hs_ds and conj_hs_ds into one dataset:
    c3s_ds = conj_hs_ds
    for conj_el in c3s_ds:
        question_id = conj_el[-1]
        one_elements = [el for el in one_hs_ds if el[-1] == question_id]
        conj_el.append(one_elements)

    n = len(c3s_ds)
    train_n = int(n * cfg.split_ratio)
    train_ds, test_ds = random_split(c3s_ds, [train_n, n - train_n])

    train_dl = DataLoader( train_ds, batch_size=train_n)

    probe = C3S(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.ccs.epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(probe, train_dl)

    torch.save(
        probe.state_dict(),
        f"{cfg.probes_dir}/truthful_qa_conj_c3s_probe.pt",
    )


def train_lr(cfg, one_hs_ds, disj_hs_ds, conj_hs_ds):
    for ds, name in [
        (one_hs_ds, "one"),
        (disj_hs_ds, "disj"),
        (conj_hs_ds, "conj"),
    ]:
        train_dl, _ = create_dataloaders([ds], batch_size=cfg.lr.batch_size)
        probe = LogisticRegression(cfg)
        trainer = pl.Trainer(
            max_epochs=cfg.lr.epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(probe, train_dl)
        torch.save(
            probe.state_dict(),
            f"{cfg.probes_dir}/truthful_qa_{name}_LR_probe.pt",
        )


if __name__ == "__main__":
    main()
