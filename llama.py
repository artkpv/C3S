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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)

seed = 42
np_rand = np.random.default_rng(seed=42)


# %%
# Load model
tokenizer = LlamaTokenizer.from_pretrained("/workspace/llama/llama-2-7b-converted")
model = LlamaForCausalLM.from_pretrained(
    "/workspace/llama/llama-2-7b-converted", 
    torch_dtype=torch.float16,
    device_map="auto"
)

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
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt", torch_dtype=torch.float16)
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=150)
pp(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# %%
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
# %%
# Measure accuracies of the probes

# %%
# Load dataset
tqa_dataset = get_tqa_dataset(np_rand)
tqa_formated_dataset_data, tqa_formated_dataset_labels = create_tokenized_tqa_dataset(
    llama.tokenizer, tqa_dataset, np_rand)

# %%  

reporter = torch.load(reporter_path, map_location=device)
