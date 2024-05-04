"""
Scripts to generate contrast pairs from the truthful_qa dataset.

Author: Artyom Karpov, www.artkpv.net
"""
from datasets import load_dataset
from jinja2 import Environment, PackageLoader, select_autoescape
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
from random import shuffle as rshuffle
import torch
from tqdm import tqdm
import os
from omegaconf import DictConfig

truthfulqa = load_dataset("truthful_qa", "generation")
env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())


# Datasets generators:
def div_items(items, max_num):
    """Return max_num items from items, taking equal number of correct and incorrect answers."""
    if max_num is not None:
        for take_correct in (True, False):
            div = max_num // 2
            rshuffle(items)
            for e in items:
                if div == 0:
                    break
                if e["is_correct"] == take_correct:
                    div -= 1
                    yield e
    else:
        yield from items


@torch.no_grad()
def one_statement_ds_generator(cfg: DictConfig):
    """Generator for a dataset with one statement per question."""

    template = env.get_template(cfg.one_template)
    counter = 0
    for row_id, row in tqdm(list(enumerate(truthfulqa["validation"])), ncols=0):
        inc_as = row["incorrect_answers"]
        cor_as = row["correct_answers"]
        counter += 1
        items = [
            {
                "id": counter,
                "question_id": row_id,
                "row": row,
                "template_render_fn": template.render,
                "is_correct": is_correct,
                "template_render_params": {
                    "question": row["question"],
                    "answer": ans,
                },
            }
            for is_correct in (True, False)
            for ans in (inc_as, cor_as)[is_correct]
        ]
        yield from div_items(items, cfg.max_samples_per_question)


@torch.no_grad()
def conj_ds_generator(cfg: DictConfig):
    """Generator for a dataset with two statements in conjunction per question."""

    template = env.get_template(cfg.conj_template)
    counter = 0
    for row_id, row in tqdm(list(enumerate(truthfulqa["validation"])), ncols=0):
        inc_as = row["incorrect_answers"]
        cor_as = row["correct_answers"]

        def create_yield(is_correct, answers):
            nonlocal counter
            counter += 1
            return {
                "id": counter,
                "question_id": row_id,
                "row": row,
                "template_render_fn": template.render,
                "is_correct": is_correct,
                "template_render_params": {
                    "question": row["question"],
                    "answers": answers,
                    "is_disjunction": False,
                },
            }

        items = (
            [
                create_yield(True, [c_a, other_c_a])
                for c_a in cor_as
                for other_c_a in [a for a in cor_as if a != c_a]
            ]
            + [create_yield(False, [i_a, c_a]) for i_a in inc_as for c_a in cor_as]
            + [create_yield(False, [c_a, i_a]) for i_a in inc_as for c_a in cor_as]
        )
        yield from div_items(items, cfg.max_samples_per_question)


@torch.no_grad()
def disj_ds_generator(cfg: DictConfig):
    """Generator for a dataset with two statements in disjunction per question."""

    template = env.get_template(cfg.disj_template)
    counter = 0
    for row_id, row in tqdm(list(enumerate(truthfulqa["validation"])), ncols=0):
        inc_as = row["incorrect_answers"]
        cor_as = row["correct_answers"]

        def create_yield(is_correct, answers):
            nonlocal counter
            counter += 1
            return {
                "id": counter,
                "question_id": row_id,
                "row": row,
                "template_render_fn": template.render,
                "is_correct": is_correct,
                "template_render_params": {
                    "question": row["question"],
                    "answers": answers,
                    "is_disjunction": True,
                },
            }

        items = (
            [create_yield(True, [c_a, i_a]) for c_a in cor_as for i_a in inc_as]
            + [
                create_yield(True, [c_a, other_c_a])
                for c_a in cor_as
                for other_c_a in [a for a in cor_as if a != c_a]
            ]
            + [
                create_yield(False, [i_a, other_i_a])
                for i_a in inc_as
                for other_i_a in [a for a in inc_as if a != i_a]
            ]
        )
        yield from div_items(items, cfg.max_samples_per_question)


# Dataset for hidden states:
@torch.no_grad()
def create_tensordataset(ds_generator, hf_model, tokenizer, cfg):
    hf_model.eval()
    items = list(ds_generator(cfg))
    neg_hs = []
    pos_hs = []
    gt_labels = []  # Ground truth labels.
    ids = []
    q_ids = []
    for item in tqdm(items, ncols=0):
        for label in (True, False):
            input_ = item["template_render_fn"](
                **item["template_render_params"], label=str(label)
            )
            t_output = tokenizer(input_, return_tensors="pt")
            t_output = {k: t_output[k].to(cfg._device) for k in t_output}
            output = hf_model(**t_output, output_hidden_states=True)
            hs = output.hidden_states[cfg.model.layer][0, -1].detach()
            if label:
                pos_hs.append(hs)
            else:
                neg_hs.append(hs)
        gt_labels.append(item["is_correct"])
        ids.append(item["id"])
        q_ids.append(item["question_id"])

    neg_hs = torch.stack(neg_hs).type(torch.float)
    pos_hs = torch.stack(pos_hs).type(torch.float)
    gt_labels = torch.tensor(gt_labels).type(torch.float)
    ids = torch.tensor(ids).type(torch.int)
    q_ids = torch.tensor(q_ids).type(torch.int)
    return TensorDataset(neg_hs, pos_hs, gt_labels, ids, q_ids)


def get_hidden_state_datasets(cfg, create_model):
    """Get hidden states datasets for one statement, disjunction and conjunction."""

    hf_model = None
    tokenizer = None
    name_gen_pairs = [
        ("one", one_statement_ds_generator),
        ("disj", disj_ds_generator),
        ("conj", conj_ds_generator),
    ]
    if ['c3s'] == cfg.probes:
        name_gen_pairs = [
            ("one", one_statement_ds_generator),
            ("conj", conj_ds_generator),
        ]
        
    for name, ds_gen in name_gen_pairs:
        ds = None
        if os.path.exists(f"{cfg.data_dir}/truthful_qa_{name}_hs_ds.pt"):
            ds = torch.load(
                f"{cfg.data_dir}/truthful_qa_{name}_hs_ds.pt", map_location=cfg._device
            )
        else:
            if hf_model is None:
                hf_model, tokenizer = create_model(cfg)
            ds = create_tensordataset(ds_gen, hf_model, tokenizer, cfg)
            torch.save(ds, f"{cfg.data_dir}/truthful_qa_{name}_hs_ds.pt")
        yield ds


def create_dataloaders(datasets, batch_size=32, split_ratio=0.8):
    """Get DataLoaders for one statement, disjunction and conjunction."""
    for ds in datasets:
        n = len(ds)
        train_n = int(n * split_ratio)
        train_ds, test_ds = random_split(ds, [train_n, n - train_n])

        yield DataLoader(
            train_ds, batch_size=train_n if batch_size == -1 else batch_size
        )
        yield DataLoader(
            test_ds, batch_size=n - train_n if batch_size == -1 else batch_size
        )
