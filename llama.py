# %%
from IPython.display import display
from collections import namedtuple
from datasets import load_dataset
from huggingface_hub import login
from jaxtyping import Int, Float
from jinja2 import Environment, PackageLoader, select_autoescape
from pathlib import Path
from pprint import pp
from torch import Tensor
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.truthful_qa_ds import get_question_answer_dataset
import copy
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.linear_model import LogisticRegression
login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)
np_rand = np.random.default_rng(seed=100500)
model_type = torch.float16

# %%
# Load model
# llama_path = "../llama/7bf_converted/"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
# tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=model_type,
    device_map=device,
)
model.eval()
pp(model)
# %%
true_token = tokenizer.encode("True")[1]
false_token = tokenizer.encode("False")[1]
print(true_token)
print(false_token)

# %%
truthfulqa = load_dataset("truthful_qa", "generation")  # 817 rows
env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())

# %%
# Playing with Tokenizer:

# %%
# qa_t = env.get_template("question_answer.jinja")
# qa_dataset = []
# for i, row in enumerate(truthfulqa["validation"]):
#     if len(row["correct_answers"]) < 2:
#         continue
#     take_correct = i % 2 == 0
#     for label in (True, False):
#         qa_dataset.append(
#             {
#                 "input": qa_t.render(
#                     row,
#                     is_correct_answer=take_correct,
#                     label=label,
#                 ),
#                 "label": label,
#                 "is_correct": take_correct,
#             }
#         )
#         correct_a = row["correct_answers"][0]
#         second_correct_a = row["correct_answers"][1]
#         incorrect_a = row["incorrect_answers"][0]
#         # qas_and_dataset.append(
#         #    {
#         #        "input": qas_t.render(
#         #            row,
#         #            a_A=correct_a if take_correct else incorrect_a,
#         #            a_B=second_correct_a,
#         #            is_disjunction=False,
#         #            label=label,
#         #        ),
#         #        "label": label,
#         #        "is_correct": take_correct,
#         #    }
#         # )
# pp(qa_dataset[0])
#
# # %%
# t_output = tokenizer(qa_dataset[0]["input"], return_tensors="pt")
#
# # %%
# pp(t_output)
# pp(len(t_output))
# pp(tokenizer.convert_ids_to_tokens(t_output["input_ids"][0, -1].item()))
# pp(tokenizer.convert_ids_to_tokens(true_token))
# pp(tokenizer.convert_ids_to_tokens(false_token))
#
# # %%
# pp(qa_dataset[0])
# pp(qa_dataset[1])
#
# # %%
# t_output = {k: t_output[k].to(device) for k in t_output}
# outputs = model(**t_output, output_hidden_states=True)
# # %%
# pred = outputs.logits[0, -2].softmax(dim=-1)
# pp(pred)
#
# # %%
# pp(
#     f"Probability of the last outputed token: {pred[t_output['input_ids'][0, -1].item()]}"
# )
# pp(f"True token probability: {pred[true_token]}")
# pp(f"False token probability: {pred[false_token]}")


# %%
# Accuracy on the TruthfulQA dataset, few shot:
count = 0
correct_samples = []
correct_n = 0
qa_t = env.get_template("question_answer.jinja")
with torch.no_grad():
    p_bar = tqdm(list(enumerate(truthfulqa["validation"])))
    for i, row in p_bar:
        def is_correct_answer(take_correct):
            input_ = qa_t.render(
                row,
                is_correct_answer=take_correct
            ),
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
            correct_samples.append(row)
        p_bar.set_description(
            f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, both {len(correct_samples)}"
        )
# Result: Correct 972, count 1634, accuracy 0.5949, both 261

# %%
# Now calculate accuracy for compound sentences for correctly 
# detected samples. Expectation: should guess all correctly.

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
                is_disjunction=False,
            ),
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

# Random? Correct 289, count 522, accuracy 0.5536, both 48: 100%|██████████| 261/261 [00:44<00:00,  5.84it/s]

# %%
def get_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["hidden_states"]
    
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def format_row(row, label, true_label):
    qa_t = env.get_template("question_answer.jinja")
    return qa_t.render(
        row,
        is_correct_answer=true_label,
        label=label,
    )


def get_hidden_states_many_examples(model, tokenizer, data, n=100):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # loop
    for i in tqdm(range(n)):
        true_label = i % 2 == 0
        # get hidden states
        neg_hs = get_hidden_states(model, tokenizer, format_row(data[i], True, true_label))
        pos_hs = get_hidden_states(model, tokenizer, format_row(data[i], False, true_label))

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(true_label)

    all_neg_hs = np.stack(all_neg_hs)
    all_pos_hs = np.stack(all_pos_hs)
    all_gt_labels = np.stack(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels

# %%
neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, truthfulqa["validation"])

# %%
# let's create a simple 50/50 train split (the data is already randomized)
n = len(y)
neg_hs_train, neg_hs_test = neg_hs[:n//2], neg_hs[n//2:]
pos_hs_train, pos_hs_test = pos_hs[:n//2], pos_hs[n//2:]
y_train, y_test = y[:n//2], y[n//2:]

# for simplicity we can just take the difference between positive and negative hidden states
# (concatenating also works fine)
x_train = neg_hs_train - pos_hs_train
x_test = neg_hs_test - pos_hs_test

lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)
print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

# %%
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
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
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
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
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
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
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
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
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
# %%
# Train CCS without any labels
ccs = CCS(neg_hs_train, pos_hs_train)
ccs.repeated_train()

# Evaluate
ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
print("CCS accuracy: {}".format(ccs_acc))
# %%
