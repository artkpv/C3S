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
imdb_ds = load_dataset('imdb')
# %%
def t5_experiments():
    samples = [
    f'''
    {imdb_ds['train']['text'][:1]}
    Did the reviewer find this movie good or bad? 
    bad
    ''',
    f'''
    {imdb_ds['train']['text'][:1]}
    Did the reviewer find this movie good or bad? 
    good
    ''',
    ]
    l11cp = torch.load('./data/allenai/unifiedqa-t5-base/imdb/quirky-neumann/reporters/layer_11.pt')

    model_name = "allenai/unifiedqa-t5-base" 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def run_model(input_string, **generator_args):
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        res = model.generate(input_ids, **generator_args)
        return tokenizer.batch_decode(res, skip_special_tokens=True)
    sample = imdb_ds['train']['text'][156:157]
    output = run_model(f'''
        {sample}
        Did the reviewer find this movie good or bad?'''
    )
    pp(f"{output['decoder_hidden_states'][0].shape=}")
    pp(f"{l11cp['norm.mean_x'].shape=}")
    #pp(samples)
    pp((output['decoder_hidden_states'][0][0,-1] == output['decoder_hidden_states'][0][1,-1]).float().mean())
    pp(l11cp['norm.mean_x'] @ output['decoder_hidden_states'][0][0,-1].T)
    pp(l11cp['norm.mean_x'] @ output['decoder_hidden_states'][0][1,-1].T)


# %%
gpt2_xl: HookedTransformer = HookedTransformer.from_pretrained("gpt2-xl")
# %%
sample = imdb_ds['train']['text'][156]
sample_false = f'{sample}\nDid the reviewer find this movie good or bad?\nGood'
sample_true = f'{sample}\nDid the reviewer find this movie good or bad?\n Bad'
with torch.inference_mode():
    _, cache_false = gpt2_xl.run_with_cache(sample_false, remove_batch_dim=True)
    _, cache_true = gpt2_xl.run_with_cache(sample_true, remove_batch_dim=True)

# %%
probe = torch.load('./data/gpt2-xl/imdb/festive-elion/reporters/layer_47.pt')
pp(probe.keys())
pp(probe['probe.0.weight'].shape)
pp(probe['probe.0.bias'].shape)
#%%
for layer in range(1, 48):
    act0 = cache_false['mlp_out', layer][-1].cpu()
    act1 = cache_true['mlp_out', layer][-1].cpu()
    p0 = act0 @ probe['probe.0.weight'].T + probe['probe.0.bias']
    p1 = act1 @ probe['probe.0.weight'].T + probe['probe.0.bias']
    confidence = 0.5*(p0 + (1-p1))
    pp(f'l {layer} {p0=} {p1=} {confidence=}')
# %%
