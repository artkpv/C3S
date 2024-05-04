"""
Scripts to evaluate probes on contrast pairs from the truthful_qa dataset.

Author: Artyom Karpov, www.artkpv.net
"""

from CCS import CCS
import logging
from tqdm import tqdm
from truthful_qa_ds import (
    one_statement_ds_generator,
    disj_ds_generator,
    conj_ds_generator,
    create_dataloaders,
    get_hidden_state_datasets,
)
from LogisticRegression import LogisticRegression
import lightning as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
import hydra
from pathlib import Path
from model import create_model

true_ids = [5574, 5852, 1565, 3009]
false_ids = [7700, 8824, 2089, 4541]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Evaluation with config: "{cfg}"')
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.probes_dir).mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg._device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {cfg._device}")

    results = pd.DataFrame(
        columns=[
            "metric_name",
            "metric_value",
            "method",
            "method_dataset",
            "known_questions_num",
            "dataset",
            "dataset_samples_num",
            "dataset_positive_num",
        ],
    )

    def add_result(**kwargs):
        logging.info(kwargs)
        results.loc[len(results)] = kwargs

    one_disj_conj_datasets = list(
        get_hidden_state_datasets(cfg, create_model=create_model)
    )

    dl_name_pairs = [
        (test_dl, name)
        for test_dl, name in zip(
            list(create_dataloaders(one_disj_conj_datasets, batch_size=-1))[1::2],
            ("one", "disj", "conj"),
        )
    ]

    with open_dict(cfg):
        cfg._input_dim = one_disj_conj_datasets[0].tensors[0].shape[-1]

    _calc_lr_accuracy(dl_name_pairs, add_result, cfg)

    _calc_ccs_and_c3s_accuracy(cfg, add_result, dl_name_pairs)

    # Random accuracy. Init CCS probe but do not train it:
    import time

    for tries in range(cfg.random_tries):
        # Set seed for pytorch to use when it creates nn.Linear:
        torch.manual_seed(int(time.time()))
        probe = CCS(cfg)
        for dl, name in dl_name_pairs:
            result = _calc_ccs_accuracy_for(probe, dl, cfg)
            result["dataset"] = name
            result["method"] = "Random"
            add_result(**result)

    # """Calculate CCS accuracy for the probe trained on all types of questions."""
    # logging.info(f"CCS probe for 'all'")
    # # Merge three Tensor datasets into one:
    # ds = torch.utils.data.ConcatDataset(one_disj_conj_datasets)
    # probe = CCS(cfg)
    # probe.load_state_dict(torch.load(f"{cfg.probes_dir}/truthful_qa_all_ccs_probe.pt"))
    # probe = probe.to(device=cfg._device)
    # for dl, name in dl_name_pairs:
    #     result = _calc_ccs_accuracy_for(probe, dl, cfg)
    #     result["dataset"] = name
    #     result["method_dataset"] = "all"
    #     add_result(**result)
    #     logging.info(results.to_string())

    results.to_csv(f"results_data.csv")
    logging.info(results.describe())
    logging.info(results.to_string())


@torch.no_grad()
def _calc_accuracy_fs_for(ds, hf_model):
    """Few-shot accuracy calculation for a given dataset generator."""

    known_questions = set()
    wrong_questions = set()
    correct_n = 0
    count = len(ds)
    pbar = tqdm(ds)
    for t_output, sample_gen in pbar:
        outputs = hf_model(**t_output, output_hidden_states=False)
        pred = outputs.logits[0, -1].softmax(dim=-1)
        token = pred.argmax(-1)
        is_correct = (
            token in true_ids if sample_gen["is_correct"] else token in false_ids
        )
        if is_correct:
            correct_n += 1
            if not sample_gen["question_id"] in wrong_questions:
                known_questions.add(sample_gen["question_id"])
        else:
            wrong_questions.add(sample_gen["question_id"])
            known_questions.discard(sample_gen["question_id"])
        pbar.set_description(
            f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, known {len(known_questions)}"
        )

    return correct_n, known_questions


def _calc_fs_accuracy(results, cfg):
    logging.info("Calculate few-shot accuracy for all types of questions.")

    fs_sets = []
    hf_model, tokenizer = create_model(cfg)

    logging.info(" ".join(f"'{tokenizer.decode(id_)}'" for id_ in true_ids))
    logging.info(" ".join(f"'{tokenizer.decode(id_)}'" for id_ in false_ids))

    for name, ds_generator in [
        ("one", one_statement_ds_generator),
        ("disj", disj_ds_generator),
        ("conj", conj_ds_generator),
    ]:
        logging.info(f"Few-shot for '{name}'")

        def map_(sample_gen):
            input_ = sample_gen["template_render_fn"](
                **sample_gen["template_render_params"], label=""
            )
            t_output = tokenizer(input_, return_tensors="pt")
            t_output = {k: t_output[k].to(cfg._device) for k in t_output}
            return t_output, sample_gen

        ds = [
            map_(sg)
            for sg in ds_generator(cfg)
        ]

        # DISABLE FS: TODO: remove it.
        ds = ds[:1]

        correct_n, known_qs = _calc_accuracy_fs_for(ds, hf_model)
        count = len(ds)
        results.loc[name, "FS, acc"] = correct_n / count
        results.loc[name, "FS, # known q."] = len(known_qs)
        results.loc[name, "FS, DS true/all"] = (
            sum(int(sg["is_correct"]) for (_, sg) in ds) / count
        )

        fs_sets.append(known_qs)
        logging.info(results.to_string())
    return fs_sets


@torch.no_grad()
def _calc_lr_accuracy_for(lr_probe, test_dl):
    """Calculate Logistic Regression accuracy for a given dataset."""
    test_neg_x, test_pos_x, test_y, _, q_ids = test_dl.dataset.dataset.tensors
    test_x = test_pos_x - test_neg_x
    y_hat = lr_probe(test_x).squeeze().sigmoid()
    y_hat = (y_hat > 0.5).float()
    accuracy = (y_hat == test_y).float().mean()
    # Get indexes of questions with all correct predictions:
    known_questions = set()
    for q_id in q_ids.unique():
        sample_indexes = (q_ids == q_id).nonzero(as_tuple=False).squeeze()
        all_guessed = (y_hat[sample_indexes] == test_y[sample_indexes]).all()
        if all_guessed:
            known_questions.add(q_id.item())

    return {
        "metric_name": "accuracy",
        "metric_value": accuracy.item(),
        "method": "LR",
        "known_questions_num": len(known_questions),
        "known_questions": known_questions,
        "dataset_samples_num": len(test_y),
        "dataset_positive_num": test_y.sum().item(),
    }


def _calc_lr_accuracy(dl_name_pairs, add_result, cfg):
    """Calculate Logistic Regression accuracy for all types of questions."""
    for dl, ds_name in dl_name_pairs:
        logging.info(f"LR probe for '{ds_name}'")
        probe = LogisticRegression(cfg).to(device=cfg._device)
        probe.load_state_dict(
            torch.load(f"{cfg.probes_dir}/truthful_qa_{ds_name}_LR_probe.pt")
        )
        result = _calc_lr_accuracy_for(probe, dl)
        result["method_dataset"] = ds_name
        result["dataset"] = ds_name
        add_result(**result)


@torch.no_grad()
def _calc_ccs_accuracy_for(ccs_probe, test_dl, cfg):
    """Calculate CCS accuracy for a given dataset."""

    # Accuracy:
    trainer = pl.Trainer()
    trainer.test(ccs_probe, test_dl, verbose=False)
    test_acc = trainer.callback_metrics["test_acc"].item()

    return {
        "metric_name": "accuracy",
        "metric_value": test_acc,
        "method": "CCS",
        "known_questions_num": len(ccs_probe.known_questions),
        "known_questions": ccs_probe.known_questions,
        "dataset_samples_num": len(test_dl.dataset.dataset.tensors[2]),
        "dataset_positive_num": test_dl.dataset.dataset.tensors[2].sum().item(),
    }


def _calc_ccs_and_c3s_accuracy(cfg, add_result, dl_name_pairs):
    """Calculate CCS accuracy for all types of questions."""
    
    probe_names =[
        f"{name}_{ptype}"
        for ptype in ['ccs', 'c3s']
        for name in ["one", "disj", "conj", 'all']
    ]
    for dl, dl_name in dl_name_pairs:
        for probe_name in probe_names:
            probe_path = Path(f"{cfg.probes_dir}/truthful_qa_{probe_name}_probe.pt")
            if not probe_path.exists():
                logging.info(f"Probe '{probe_name}' not found, skip.")
                continue
            logging.info(f"CCS probe for '{probe_name}' probe, {dl_name} dataset")
            probe = CCS(cfg)
            probe.load_state_dict( torch.load(probe_path))
            probe = probe.to(device=cfg._device)
            result = _calc_ccs_accuracy_for(probe, dl, cfg)
            result["dataset"] = dl_name
            result["method_dataset"] = probe_name
            add_result(**result)


def intersection(a, b):
    """Calculate intersection matrix for two sets of sets."""
    im = np.zeros((3, 3))
    for i, a_set in enumerate(a):
        for j, b_set in enumerate(b):
            im[i, j] = len(a_set.intersection(b_set))
    return im


def _analyse_sets(fs_sets, lr_sets, ccs_sets):
    """Analyse sets of known questions."""
    sets_names = ["FS", "CCS", "LR"]
    subsets_names = ["One statement", "Disjunction", "Conjunction"]
    for i, sets in enumerate([fs_sets, ccs_sets, lr_sets]):
        for j, sets2 in enumerate([fs_sets, ccs_sets, lr_sets][i + 1 :]):
            j = j + i + 1
            logging.info(f"Intersection {sets_names[i]} vs {sets_names[j]}:")
            df = pd.DataFrame(
                intersection(sets, sets2),
                columns=subsets_names,
                index=subsets_names,
            )
            logging.info(df)
            np.save(
                f"truthful_qa_{sets_names[i]}_vs_{sets_names[j]}.npy",
                df.to_numpy(),
            )

    ccs_all_qs = set.union(*ccs_sets)
    fs_all_qs = set.union(*fs_sets)
    lr_all_qs = set.union(*lr_sets)

    # Questions identified by CCS but not by LR or by FS:
    ccs_only_qs = ccs_all_qs - fs_all_qs - lr_all_qs
    # Questions identified by LR but not by CCS or by FS:
    lr_only_qs = lr_all_qs - fs_all_qs - ccs_all_qs
    # Questions identified by FS but not by CCS or by LR:
    fs_only_qs = fs_all_qs - ccs_all_qs - lr_all_qs

    # Display:
    logging.info("Unique questions identified by CCS but not by LR or by FS:")
    df = pd.DataFrame(
        {
            "CCS only": len(ccs_only_qs),
            "LR only": len(lr_only_qs),
            "FS only": len(fs_only_qs),
            "CCS and LR": len(ccs_all_qs & lr_all_qs),
            "CCS and FS": len(ccs_all_qs & fs_all_qs),
            "LR and FS": len(lr_all_qs & fs_all_qs),
            "CCS and LR and FS": len(ccs_all_qs & lr_all_qs & fs_all_qs),
        },
        index=["# questions"],
    ).T

    logging.info(df)


if __name__ == "__main__":
    main()
