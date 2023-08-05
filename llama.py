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
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import einops
import elk 
from pathlib import Path

import circuitsvis as cv
#from promptsource.templates import DatasetTemplates
from plotly_utils import imshow
from functools import cache
from utils.datasets import get_tqa_dataset, create_tokenized_tqa_dataset


from huggingface_hub import login
login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)

seed = 42
np_rand = np.random.default_rng(seed=42)


# %%
# Load model
tokenizer = LlamaTokenizer.from_pretrained("/workspace/llama/7Bf_converted")
model = LlamaForCausalLM.from_pretrained(
    "/workspace/llama/7Bf_converted", 
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# %%
#llama = HookedTransformer.from_pretrained(
#    "Llama-2-7b",
#    hf_model=model, 
#    device='cpu', 
#    fold_ln=False, 
#    center_writing_weights=False, 
#    center_unembed=False, 
#    tokenizer=tokenizer
#)
# pp(llama) 
# pp(llama.generate("The capital of Germany is", max_new_tokens=20, temperature=0))

# %%
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer
)
prompt= '''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
'''
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# %% 
prompt = "Write binary search algorithm in Python. Answer:"
batch = tokenizer(prompt, return_tensors="pt")
batch = {k: v.to("cuda") for k, v in batch.items()}
with torch.no_grad():
    outputs = model.generate(**batch, max_length=150)
output_text = tokenizer.decode(outputs[0])
pp(output_text)


# %%
# Measure accuracies of the probes

# %%
# Load dataset
tqa_dataset = get_tqa_dataset(np_rand)
# %%
tqa_formated_dataset_data, tqa_formated_dataset_labels = create_tokenized_tqa_dataset(
    tokenizer, tqa_dataset, np_rand)

# %%  
reporter_path = Path('/workspace/llama/7Bf_converted/dbpedia_14/gifted-poitras/reporters/layer_31.pt')
reporter = torch.load(
    reporter_path,
    map_location=device
).to(torch.float16)

# %%
ids = tqa_formated_dataset_data[0]
with torch.no_grad():
    outputs = model(
        ids.reshape((1,-1)).to(device),
        output_hidden_states=True
    )

val_credences = reporter(outputs.hidden_states[31][0])
# %%
