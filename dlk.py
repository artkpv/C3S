#%%

from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from pprint import pp
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
# %%
l11cp = torch.load('./artifacts/allenai/unifiedqa-t5-base/imdb/quirky-neumann/reporters/layer_11.pt')

# %%
ds = load_dataset('imdb')
# %%
samples = [
f'''
{ds['train']['text'][:1]}
Did the reviewer find this movie good or bad? 
bad
''',
f'''
{ds['train']['text'][:1]}
Did the reviewer find this movie good or bad? 
good
''',
]
# %%

model_name = "allenai/unifiedqa-t5-base" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
# %%
sample = ds['train']['text'][156:157]
res = run_model(
f'''
{sample}
Did the reviewer find this movie good or bad? 
''')
# %%
pp(sample)
pp(res)


# %%
pp(f"{output['decoder_hidden_states'][0].shape=}")
pp(f"{l11cp['norm.mean_x'].shape=}")
#pp(samples)
# %%
pp((output['decoder_hidden_states'][0][0,-1] == output['decoder_hidden_states'][0][1,-1]).float().mean())
# %%
pp(l11cp['norm.mean_x'] @ output['decoder_hidden_states'][0][0,-1].T)
pp(l11cp['norm.mean_x'] @ output['decoder_hidden_states'][0][1,-1].T)
# %%
# %%
gpt2_xl: HookedTransformer = HookedTransformer.from_pretrained("gpt2-xl")
# %%
