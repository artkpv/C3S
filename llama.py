#%%
from IPython.display import display
from tqdm import tqdm
import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int, Float
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
from pprint import pp
#from transformer_lens.hook_points import HookPoint
#from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from pathlib import Path

#from promptsource.templates import DatasetTemplates
from utils.truthful_qa_ds import get_question_answer_dataset


from huggingface_hub import login
login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)

seed = 42
np_rand = np.random.default_rng(seed=42)
model_type = torch.float16

# 

from datasets import load_dataset
from jinja2 import Environment, PackageLoader, select_autoescape
import torch

# %%
def get_question_answer_dataset():
    """
    Returns dataset with prompts for single answer, for disjunction and for conjunction.
    """
    truthfulqa = load_dataset("truthful_qa", "generation")  # 817 rows
    env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())
    qa_t = env.get_template("question_answer.jinja")
    qas_t = env.get_template("question_answers.jinja")
    qas_l_t = env.get_template("question_answers_one_line.jinja")
    qa_dataset = []
    qas_and_dataset = []
    for i, row in enumerate(truthfulqa["validation"]):
        if len(row["correct_answers"]) < 2:
            continue
        take_correct = i%2==0
        for label in (True, False):
            qa_dataset.append(
                {
                    "input": qa_t.render(
                        row,
                        is_correct_answer=take_correct,
                        label=label,
                    ),
                    "label": label,
                    "is_correct": take_correct,
                }
            )
            correct_a = row["correct_answers"][0]
            second_correct_a = row["correct_answers"][1]
            incorrect_a = row["incorrect_answers"][0]
            #qas_and_dataset.append(
            #    {
            #        "input": qas_t.render(
            #            row,
            #            a_A=correct_a if take_correct else incorrect_a,
            #            a_B=second_correct_a,
            #            is_disjunction=False,
            #            label=label,
            #        ),
            #        "label": label,
            #        "is_correct": take_correct,
            #    }
            #)
    return qa_dataset, qas_and_dataset


# %%

qa_dataset, qas_and_dataset = get_question_answer_dataset()
#pp(qas_and_dataset[0])
pp(qa_dataset[0])

# %%
# Load model
#llama_path = "../llama/7bf_converted/"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
    
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf"
)
model.eval()
pp(model)

# %%
true_token = tokenizer.encode('True')[1]
false_token = tokenizer.encode('False')[1]
print(true_token)
print(false_token)

#%%
t_output = tokenizer(qa_dataset[0]['input'], return_tensors="pt")

# %%
pp(t_output)
pp(len(t_output))
pp(tokenizer.convert_ids_to_tokens(t_output['input_ids'][0, -1].item()))
pp(tokenizer.convert_ids_to_tokens(true_token))
pp(tokenizer.convert_ids_to_tokens(false_token))

#%%
pp(qa_dataset[0])
pp(qa_dataset[1])

#%%
outputs = model(**t_output, output_hidden_states=True)
# %%
pred = outputs.logits[0, -2].softmax(dim=-1)
pp(pred)

#%% 
pp(f"Probability of the last outputed token: {pred[t_output['input_ids'][0, -1].item()]}")
pp(f'True token probability: {pred[true_token]}')
pp(f'False token probability: {pred[false_token]}')

# %%
# Accuracy on the TruthfulQA dataset:
correct = 0
count = 100
for sample in tqdm(qa_dataset[:count]):
    t_output = tokenizer(sample['input'], return_tensors="pt")
    outputs = model(**t_output, output_hidden_states=False)
    pred = outputs.logits[0, -2].softmax(dim=-1)
    true_prob = pred[true_token]
    false_prob = pred[false_token]
    is_true = true_prob > false_prob
    correct += is_true == sample['is_correct']
print(f"Correct {correct}, count {count}, accuracy {correct / count:.4}")

#correct = 0
#batch_size = 10
#for start_i in tqdm(range(0, 1000, batch_size)):
#    batch = qa_dataset[start_i:start_i + batch_size]
#    tokenized = [
#        tokenizer(t['input'], padding='max_length', truncation=False, return_tensors="pt")
#        for t in batch
#    ]
#    input_ids = torch.stack([torch.tensor(el['input_ids']) for el in tokenized])
#    masks = torch.stack([torch.tensor(el['attention_mask']) for el in tokenized])
#    outputs = model(
#        input_ids=input_ids,
#        attention_masks=masks, 
#        output_hidden_states=False)
#    pred = outputs.logits[:, -2].softmax(dim=-1)
#    true_prob = pred[:, true_token]
#    false_prob = pred[:, false_token]
#    is_true = true_prob > false_prob
#    correct += torch.sum(is_true == torch.tensor([s['is_correct'] for s in batch])).item()
#print(f"Accuracy: {correct * 1.0 / len(qa_dataset) * 100.0 :.2}%")



# %%
# Playing with LLAMA-2

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=model_type,
#     device_map="auto",
#     tokenizer=tokenizer
# )
# prompt= '''
# <s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
# 
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>
# 
# There's a llama in my garden 😱 What should I do? [/INST]
# '''
# sequences = pipeline(
#     prompt,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=500,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
# 

# %% 
prompt = "Write binary search algorithm in Python. Answer:"
batch = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**batch, max_new_tokens=100)[0]
pp(tokenizer.decode(outputs, skip_special_tokens=True))


# %%
# Measure accuracies of the probes

# %%
# Load dataset
#tqa_dataset = get_tqa_dataset(np_rand)
## %%
#tqa_formated_dataset_data, tqa_formated_dataset_labels = create_tokenized_tqa_dataset(
#    tokenizer, tqa_dataset, np_rand)
#
## %%  
#reporter_path = Path('data/llama-7bf/dbpedia_14/gifted-poitras/reporters/layer_31.pt')
##reporter_path = Path('/workspace/llama/7Bf_converted/dbpedia_14/unruffled-margulis/reporters/layer_31.pt')
#reporter = torch.load(
#    reporter_path,
#    map_location='cpu'
#)

# %%
def predict(sampleid):
    with torch.inference_mode():
        outputs = model(
            tqa_formated_dataset_data[sampleid].to(device).reshape((1,-1)),
            output_hidden_states=True
        )
        activations = outputs.hidden_states[31].to('cpu').to(torch.float32).squeeze()
        pp(f'{activations.shape=}')

        labels = tqa_formated_dataset_labels[sampleid]
        pp(labels)
        token_ids = [e[0] for e in labels]
        pp(f'{token_ids=}')
        r_out = reporter(activations[token_ids])
        pp(r_out)

predict(sampleid = 0)

# %%
