# %%
# from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML
from collections import namedtuple
from datasets import load_dataset
from huggingface_hub import login
from jaxtyping import Int, Float
from jinja2 import Environment, PackageLoader, select_autoescape
from pathlib import Path
from pprint import pp
from torch import Tensor
from tqdm import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.truthful_qa_ds import get_question_answer_dataset
import copy
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens.utils as utils
import transformers
import circuitsvis as cv
from einops import einsum

# %%
login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)
np_rand = np.random.default_rng(seed=100500)
model_type = torch.float16

# %%
# Part 1. Calculate accuracies.
# =============================

# %%
# Load model
tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    # device_map=device,
)
tokenizer.add_special_tokens({"pad_token": "<pad>"})
# tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=model_type, device_map=device
)

# %%
model.eval()
pp(model)
pp(model.config)
# %%
with torch.no_grad():
    pp(
        tokenizer.batch_decode(
            model.generate(
                tokenizer("The capital of Russia is", return_tensors="pt").input_ids.to(
                    device
                ),
                max_length=20,
            )
        )[0]
    )

# %%
true_token = tokenizer.encode("True")[1]
false_token = tokenizer.encode("False")[1]
print(true_token)
print(false_token)

# %%
truthfulqa = load_dataset("truthful_qa", "generation")  # 817 rows
env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())


# %%
# Accuracy on the TruthfulQA dataset, few shot.


def calc_accuracy_for_one_statement():
    correct_samples = []
    count = 0
    correct_n = 0
    qa_t = env.get_template("question_answer.jinja")
    with torch.no_grad():
        p_bar = tqdm(list(enumerate(truthfulqa["validation"])))
        for i, row in p_bar:

            def is_correct_answer(take_correct):
                input_ = qa_t.render(row, is_correct_answer=take_correct, label="")
                t_output = tokenizer(input_, return_tensors="pt")
                t_output = {k: t_output[k].to(device) for k in t_output}
                outputs = model(**t_output, output_hidden_states=False)
                pred = outputs.logits[0, -1].softmax(dim=-1)
                token = pred.argmax(-1)
                is_correct = (
                    token == true_token if take_correct else token == false_token
                )
                return is_correct
                # predicted = (pred[true_token] > pred[false_token]).item()
                # return predicted == take_correct

            with_true = is_correct_answer(True)
            count += 1
            if with_true:
                correct_n += 1
            with_false = is_correct_answer(False)
            count += 1
            if with_false:
                correct_n += 1
            if with_true and with_false:
                correct_samples.append(row)
            p_bar.set_description(
                f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, both {len(correct_samples)}"
            )
    return correct_samples


correct_samples = calc_accuracy_for_one_statement()

# Result for true_token > false_token : Correct 972, count 1634, accuracy 0.5949, both 261
# Result for predicted token == true_token or false_token: Correct 974, count 1634, accuracy 0.5961, both 263: 100%|██████████| 817/817 [01:21<00:00, 10.04it/s]


# %%
# Accuracy for disjunction / conjunction sentences for correctly
# detected samples. Few shot.
def calc_accuracy_for(is_disjunction):
    count = 0
    correct_compound = []
    correct_n = 0
    qas_t = env.get_template("question_answers.jinja")
    with torch.no_grad():
        p_bar = tqdm(list(enumerate(correct_samples)))
        for i, row in p_bar:

            def is_correct_answer(take_correct):
                input_ = qas_t.render(
                    row,
                    is_correct_answer=take_correct,
                    is_disjunction=is_disjunction,
                    label="",
                )
                t_output = tokenizer(input_, return_tensors="pt")
                t_output = {k: t_output[k].to(device) for k in t_output}
                outputs = model(**t_output, output_hidden_states=False)
                pred = outputs.logits[0, -1].softmax(dim=-1)
                predicted = (pred[true_token] > pred[false_token]).item()
                return predicted == take_correct

            with_true = is_correct_answer(True)
            count += 1
            if with_true:
                correct_n += 1
            with_false = is_correct_answer(False)
            count += 1
            if with_false:
                correct_n += 1
            if with_true and with_false:
                correct_compound.append(row)
            p_bar.set_description(
                f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, both {len(correct_compound)}"
            )


# %%
calc_accuracy_for(is_disjunction=True)
# Disjunction (OR):
# Correct 331, count 522, accuracy 0.6341, both 72: 100%|██████████| 261/261 [00:58<00:00,  4.50it/s]

# %%
calc_accuracy_for(is_disjunction=False)
# Conjunction (AND):
# Correct 289, count 522, accuracy 0.5536, both 48: 100%|██████████| 261/261 [00:44<00:00,  5.84it/s]


# %%
# Get hidden states for probes
VERBOSE = False


def get_hidden_states_many_examples(
    model,
    tokenizer,
    data,
    n=100,
    template="question_answer.jinja",
    is_disjunction=False,
):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels

    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """

    def get_hidden_states(model, tokenizer, input_text, layer=-1):
        """
        Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last)
        on that input text (where the full text is given to the encoder).
        Returns a numpy array of shape (hidden_dim,)
        """
        if VERBOSE:
            print("=" * 10)
            print(input_text)
            print("=" * 10)
        encoder_text_ids = tokenizer(
            input_text, truncation=True, return_tensors="pt"
        ).input_ids.to(model.device)
        with torch.no_grad():
            output = model(encoder_text_ids, output_hidden_states=True)
        hs_tuple = output["hidden_states"]
        hs = hs_tuple[layer][0, -1].detach()
        return hs

    def format_row(row, label, true_label, template, is_disjunction):
        env_t = env.get_template(template)
        return env_t.render(
            row,
            is_correct_answer=true_label,
            label=str(label),
            is_disjunction=is_disjunction,
        )

    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # loop
    for i in tqdm(range(n)):
        true_label = i % 2 == 0
        # get hidden states
        neg_hs = get_hidden_states(
            model,
            tokenizer,
            format_row(
                data[i],
                label=True,
                true_label=true_label,
                template=template,
                is_disjunction=is_disjunction,
            ),
        )
        pos_hs = get_hidden_states(
            model,
            tokenizer,
            format_row(
                data[i],
                label=False,
                true_label=true_label,
                template=template,
                is_disjunction=is_disjunction,
            ),
        )

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(torch.tensor(true_label).to(device))

    all_neg_hs = torch.stack(all_neg_hs).type(torch.float)
    all_pos_hs = torch.stack(all_pos_hs).type(torch.float)
    all_gt_labels = torch.stack(all_gt_labels).type(torch.float)

    return all_neg_hs, all_pos_hs, all_gt_labels


def get_hs_train_test_ds(n=800, template="question_answer.jinja", is_disjunction=False):
    neg_hs, pos_hs, y = get_hidden_states_many_examples(
        model,
        tokenizer,
        truthfulqa["validation"],
        n=n,
        template=template,
        is_disjunction=is_disjunction,
    )
    n = len(y)
    train_num = int(n * 0.8)
    neg_hs_train, neg_hs_test = neg_hs[:train_num], neg_hs[train_num:]
    pos_hs_train, pos_hs_test = pos_hs[:train_num], pos_hs[train_num:]
    y_train, y_test = y[:train_num], y[train_num:]
    return neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test


def convert_to_difference_hs_train_test_ds(
    neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test
):
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test
    return x_train, y_train, x_test, y_test


# %%
# Test
VERBOSE = True
get_hs_train_test_ds(template="question_answers.jinja", is_disjunction=False, n=5)
get_hs_train_test_ds(template="question_answers.jinja", is_disjunction=True, n=5)
VERBOSE = False

# %%
# Dataset
NUM = 800
hs_ds = get_hs_train_test_ds(n=NUM)
hs_qans_conj_ds = get_hs_train_test_ds(
    template="question_answers.jinja", is_disjunction=False, n=NUM
)
hs_qans_disj_ds = get_hs_train_test_ds(
    template="question_answers.jinja", is_disjunction=True, n=NUM
)

diff_ds = convert_to_difference_hs_train_test_ds(*hs_ds)
diff_qans_conj_ds = convert_to_difference_hs_train_test_ds(*hs_qans_conj_ds)
diff_qans_disj_ds = convert_to_difference_hs_train_test_ds(*hs_qans_disj_ds)


# %%
# Logigictic regression
class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.fc(x).squeeze()
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def calc_LR_accuracy(x_train, y_train, x_test, y_test):
    x_train = x_train.to("cpu")
    x_test = x_test.to("cpu")
    y_train = y_train.to("cpu")
    y_test = y_test.to("cpu")
    LR_probe = LogisticRegression(x_train.shape[-1])
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(
        LR_probe,
        DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True),
    )
    LR_probe.eval()
    # Accuracy:
    y_hat = LR_probe(x_test).squeeze().sigmoid()
    y_hat = (y_hat > 0.5).float()
    acc = (y_hat == y_test).float().mean()
    print("Logistic regression accuracy: {}".format(acc))
    return LR_probe


# LR_probe = LogisticRegression(class_weight="balanced")
# def calc_LR_accuracy(x_train, y_train, x_test, y_test):
#    LR_probe.fit(x_train, y_train)
#    print("Logistic regression accuracy: {}".format(LR_probe.score(x_test, y_test)))

# %%
print("One statement")
statement_LR_probe = calc_LR_accuracy(*diff_ds)
# Logistic regression accuracy: 0.831250011920929

# %%
print("Disjunction statement")
disj_LR_probe = calc_LR_accuracy(*diff_qans_disj_ds)

# Logistic regression accuracy: 0.7749999761581421

# %%
print("Conjunction statement")
conj_LR_probe = calc_LR_accuracy(*diff_qans_conj_ds)

# Logistic regression accuracy: 0.8187500238418579


# %%
# CCS probe
class CCS(object):
    def __init__(
        self,
        x0,
        x1,
        nepochs=1000,
        ntries=10,
        lr=1e-3,
        batch_size=-1,
        verbose=False,
        device="cuda",
        weight_decay=0.01,
        var_normalize=False,
    ):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        self.probe.to(self.device)

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(
            self.x0, dtype=torch.float, requires_grad=False, device=self.device
        )
        x1 = torch.tensor(
            self.x1, dtype=torch.float, requires_grad=False, device=self.device
        )
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1) ** 2).mean(0)
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(
            self.normalize(x0_test),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        x1 = torch.tensor(
            self.normalize(x1_test),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        with torch.no_grad():
            p0 = self.best_probe(x0)
            p1 = self.best_probe(x1)
        avg_confidence = 0.5 * (p0 + (1 - p1))
        predictions = (avg_confidence.detach() < 0.5).int()[:, 0]
        acc = (predictions == y_test).float().mean()
        acc = max(acc, 1 - acc)

        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        epoch_j = list(
            (epoch, j) for epoch in range(self.nepochs) for j in range(nbatches)
        )

        for epoch, j in epoch_j:
            x0_batch = x0[j * batch_size : (j + 1) * batch_size]
            x1_batch = x1[j * batch_size : (j + 1) * batch_size]

            # probe
            p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.detach().cpu().item()

    def repeated_train(self):
        best_loss = np.inf
        pbar = tqdm(list(range(self.ntries)))
        for train_num in pbar:
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
            pbar.set_description(f"Train {train_num}. Best loss {best_loss:.4}")

        return best_loss


# %%
def calc_random_probe_and_ccs_accuracies(
    neg_hs_train,
    pos_hs_train,
    y_train,
    neg_hs_test,
    pos_hs_test,
    y_test,
    lr=1e-3,
    batch_size=-1,
    nepocs=1000,
    random_tries=10,
):
    rand_accuracies = []
    best_rand_acc_probe = None
    best_rand_acc = 0.0
    for t in range(random_tries):
        ccs = CCS(neg_hs_train, pos_hs_train, lr=lr, nepochs=nepocs)
        rand_accuracies.append(ccs.get_acc(neg_hs_test, pos_hs_test, y_test).item())
        if rand_accuracies[-1] > best_rand_acc:
            best_rand_acc = rand_accuracies[-1]
            best_rand_acc_probe = ccs
    rand_accuracies = np.array(rand_accuracies)

    ccs = best_rand_acc_probe
    ccs.repeated_train()
    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    return ccs, rand_accuracies.mean(), ccs_acc, rand_accuracies.std()


# %%
# Do sweep to find better LR and BS for CCS probe:
names = [
    "One statement",
    "Disjunction",
    "Conjunction",
]
best_probes = [
    (None, 0, 0),
    (None, 0, 0),
    (None, 0, 0),
]
for lr in (1e-3, 1e-4, 1e-5):
    for bs in (32, 128, 512, -1):
        print(f"lr={lr}, bs={bs}")
        for i, ds in enumerate([hs_ds, hs_qans_disj_ds, hs_qans_conj_ds]):
            probe, rand_acc, ccs_acc, _ = calc_random_probe_and_ccs_accuracies(
                *ds, lr=lr, batch_size=bs, nepocs=50
            )
            if ccs_acc > best_probes[i][2]:
                best_probes[i] = (probe, rand_acc, ccs_acc)
                print(
                    f"{names[i]}. Best CCS accuracy: {best_probes[i][2]:.4}, random accuracy: {best_probes[i][1]:.4}, lr={lr}, bs={bs}"
                )

"""
Result of the sweep:
One statement. Best CCS accuracy: 0.825, random accuracy: 0.7312, lr=0.001, bs=32
Disjunction. Best CCS accuracy: 0.55, random accuracy: 0.525, lr=0.001, bs=32
Conjunction. Best CCS accuracy: 0.5938, random accuracy: 0.5688, lr=0.001, bs=32
Disjunction. Best CCS accuracy: 0.5875, random accuracy: 0.575, lr=0.001, bs=-1
Conjunction. Best CCS accuracy: 0.6313, random accuracy: 0.6625, lr=0.0001, bs=32
Conjunction. Best CCS accuracy: 0.6562, random accuracy: 0.5875, lr=1e-05, bs=128
Conjunction. Best CCS accuracy: 0.675, random accuracy: 0.5875, lr=1e-05, bs=-1
"""

# %%
# Calc accuracy for random and CCS probes:
probes = []
for i, ds in enumerate([hs_ds, hs_qans_disj_ds, hs_qans_conj_ds]):
    probe, rand_acc_mean, ccs_acc, rand_acc_std = calc_random_probe_and_ccs_accuracies(
        *ds, lr=1e-4, batch_size=128, nepocs=1000, random_tries=200
    )
    probes.append((probe, rand_acc, ccs_acc))
    print(
        f"""{names[i]}. 
        Best CCS accuracy: {ccs_acc:.4}
        Random accuracy: {rand_acc_mean:.4} mean, {rand_acc_std:.4} std
        """
    )

# %%
# Save probes:
for i, (probe, rand_acc, ccs_acc) in enumerate(probes):
    torch.save(
        probe.best_probe.state_dict(),
        f"data/llama-probes/truthful_qa/ccs_{names[i].lower().replace(' ', '_')}_probe.pt",
    )

torch.save(
    statement_LR_probe.state_dict(),
    f"data/llama-probes/truthful_qa/statement_LR_probe_probe.pt",
)
torch.save(
    disj_LR_probe.state_dict(),
    f"data/llama-probes/truthful_qa/disj_LR_probe.pt",
)
torch.save(
    conj_LR_probe.state_dict(),
    f"data/llama-probes/truthful_qa/conj_LR_probe.pt",
)

# %%
# ==========
# Part 2. MI
# ==========

# %%
tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    # device_map=device,
)
tokenizer.add_special_tokens({"pad_token": "<pad>"})
# tokenizer.pad_token = tokenizer.eos_token

hf_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    # torch_dtype=model_type,
    # device_map=device,
    low_cpu_mem_usage=True,
)

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    hf_model=hf_model,
    device="cpu",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=tokenizer,
    dtype=model_type,
    use_attn_result=True,
    use_split_qkv_input=True,
)
model = model.to(device)
model.eval()
pp(model)
pp(model.cfg)
model.generate("The capital of China is", max_new_tokens=20, temperature=0)

# %%
truthfulqa = load_dataset("truthful_qa", "generation")  # 817 rows
env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())

# %%
# Load probes:
d_model = model.cfg.d_model
LR_probes = []

for p_paths in [
    "data/llama-probes/truthful_qa/statement_LR_probe_probe.pt",
    f"data/llama-probes/truthful_qa/disj_LR_probe.pt",
    f"data/llama-probes/truthful_qa/conj_LR_probe.pt",
]:
    probe = LogisticRegression(d_model)
    probe.load_state_dict(torch.load(p_paths))
    probe.eval()
    LR_probes.append(probe)
    pp(probe)

names = [
    "One statement",
    "Disjunction",
    "Conjunction",
]
ccs_probes = []
for i, name in enumerate(names):
    probe = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
    loaded = torch.load(
        f"data/llama-probes/truthful_qa/ccs_{names[i].lower().replace(' ', '_')}_probe.pt",
    )
    probe.load_state_dict(loaded)
    probe.eval()
    ccs_probes.append(probe)
    pp(probe)

# %%
# Get examples
# One sentence
row_id = 125
row = truthfulqa["validation"][row_id]
env_t = env.get_template("question_answer.jinja")
input_text = env_t.render(row, is_correct_answer=True, label=str(False))
with torch.no_grad():
    logits, cache = model.run_with_cache(input_text)
# Last layer hidden states:
final_residual_stream = cache["resid_post", -1]
assert d_model == final_residual_stream.shape[-1]

# %%
# Visualising Attention Heads
attention_pattern = cache["pattern", 0, "attn"]
print(attention_pattern.shape)
str_tokens = model.to_str_tokens(input_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(
    tokens=str_tokens, 
    attention=attention_pattern,
    #attention_head_names=[f"L0H{i}" for i in range(12)],   # Breaks for me.
)
# %%
