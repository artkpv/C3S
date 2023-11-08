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

def format_imdb(text, label):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "The following movie review expresses a " + ["negative", "positive"][label] + " sentiment:\n" + text


def get_hidden_states_many_examples(model, tokenizer, data, model_type, n=100):
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
    for _ in tqdm(range(n)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        while True:
            idx = np.random.randint(len(data))
            text, true_label = data[idx]["content"], data[idx]["label"]
            # the actual formatted input will be longer, so include a bit of a marign
            if len(tokenizer(text)) < 400:  
                break
                
        # get hidden states
        neg_hs = get_hidden_states(model, tokenizer, format_imdb(text, 0), model_type=model_type)
        pos_hs = get_hidden_states(model, tokenizer, format_imdb(text, 1), model_type=model_type)

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(true_label)

    all_neg_hs = np.stack(all_neg_hs)
    all_pos_hs = np.stack(all_pos_hs)
    all_gt_labels = np.stack(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels

# %%
neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data, model_type)