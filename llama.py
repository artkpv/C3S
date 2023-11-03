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
def get_samples(template):
    qa_t = env.get_template(template)
    result = []
    for i, row in enumerate(truthfulqa["validation"]):
        if len(row["correct_answers"]) < 2:
            continue
        take_correct = i % 2 == 0
        for label in (True, False):
            result.append({
                "index": i,
                "row": row,
                "input": qa_t.render(
                    row,
                    is_correct_answer=take_correct,
                    label=label,
                ),
                "label": label,
                "is_correct": take_correct,
            })
    return result

# %%
# Accuracy on the TruthfulQA dataset:
count = 0
true_count = 0
correct_samples = []
with torch.no_grad():
    for sample in tqdm(get_samples("question_answer.jinja")[:1000]):
        count += 1
        t_output = tokenizer(sample["input"], return_tensors="pt")
        t_output = {k: t_output[k].to(device) for k in t_output}
        outputs = model(**t_output, output_hidden_states=False)
        pred = outputs.logits[0, -2].softmax(dim=-1)
        true_prob = pred[true_token]
        false_prob = pred[false_token]
        is_true = true_prob > false_prob
        true_count += int(sample["label"])
        if is_true == sample["is_correct"]:
            correct_samples.append(sample['row'])

# %%
correct_n = len(correct_samples)
print(
    f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, true label count {true_count}"
)

# %%
# Calculate accuracy for correctly detectedly samples.
qas_correct_samples = []
qas_t = env.get_template("question_answers.jinja")
for i, row in tqdm(enumerate(correct_samples)):
    take_correct = i % 2 == 0
    for label in (True, False):
        input_ = qas_t.render(
            row,
            is_disjunction=True,
            is_correct_answer=take_correct,
            label=label,
        ),
        t_output = tokenizer(input_, return_tensors="pt")
        t_output = {k: t_output[k].to(device) for k in t_output}
        outputs = model(**t_output, output_hidden_states=False)
        pred = outputs.logits[0, -2].softmax(dim=-1)
        true_prob = pred[true_token]
        false_prob = pred[false_token]
        is_true = true_prob > false_prob
        true_count += int(label)
        if is_true == take_correct:
            qas_correct_samples.append((row, input_))

correct_n = len(qas_correct_samples)
print(
    f"Correct {correct_n}, count {count}, accuracy {correct_n / count:.4}, true label count {true_count}"
)

# %%
