'''
For GPT2-XL model.
'''
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
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Model, GPT2Tokenizer, \
    DebertaV2Model, DebertaV2Tokenizer, LlamaForCausalLM, LlamaTokenizer
from sklearn.linear_model import LogisticRegression
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
# GPT2-XL from HF
gpt2_xl : GPT2Model = GPT2Model.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl.eval()
pp(gpt2_xl)

# %%
# Creates a multi-sentence dataset based on TruthfulQA dataset (in each 
# sample there are many question and answer pairs).
tqa_dataset = get_tqa_dataset(np_rand)
tqa_formated_dataset_data, tqa_formated_dataset_labels = create_tokenized_tqa_dataset(
    tokenizer, tqa_dataset, np_rand)

# %% 
#  Calculates accuracy on this dataset for GPT2-xl model and a probe.

layer=47
#dataset_name = 'dbpedia_14'
#reporter = elk.training.Reporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
#reporter = torch.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
#reporter = torch.load(f'./data/deberta-v2-xxlarge-mnli/imdb/eager-colden/reporters/layer_{layer}.pt', map_location=device)
#reporter = elk.training.CcsReporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
# pp(reporter.__dict__.keys())
# #reporter.eval()
# correct_num = 0
# all_num = 0
# # TODO: vectorized version. Now it runs like forever.
# SOME_SUBSET=10
# for data, labels in tqdm(list(zip(tqa_formated_dataset_data, tqa_formated_dataset_labels))[:SOME_SUBSET]):
#     with torch.inference_mode():
#         output = deberta.forward(
#             data.to(device),
#             output_hidden_states=True
#         )
#         h_states = output['hidden_states'][layer][0].to(device)
#         logits = reporter(h_states)
#         pp(logits.shape)
#         res = logits.sigmoid()
#         #visualize(data, tokenizer, res, labels)
#         for (pos, l) in labels:
#             all_num += 1
#             if (res[pos-1] > 0.5) == (l == 1):
#                 correct_num += 1
#     print(f'''Accuracy for TruthfulQA, using GPT2-xl, and probe from {dataset_name} on {layer}:
# {correct_num}/{all_num} = {correct_num/all_num:.2}.''')

#%%
# Investigates an example from this dataset.
with torch.inference_mode():
    output = gpt2_xl.forward(
        tqa_formated_dataset_data[0],
        output_hidden_states=True, 
        output_attentions=True,
    )
cache_true = output['hidden_states']
layers_num = len(output['hidden_states'])
heads_num = output['attentions'][1].size(1)
pp(f'{layers_num=} {heads_num=}')
pp(type(output))
pp(output.keys())
pp(output['last_hidden_state'].shape)
pp(len(output['hidden_states']))
pp(len(output['attentions']))
pp(len(output['past_key_values']))
pp('hidden states:')
for i,e in enumerate(cache_true):
    pp(f'{i} {e.shape}')
pp('attentions:')
for i,e in enumerate(output['attentions']):
    pp(f'{i} {e.shape}')
pp(output['last_hidden_state'].shape)
# See https://huggingface.co/docs/transformers/main_classes/output

# %%
# Visualizing scores per tokens
def visualize(layer, reporter):
    with torch.inference_mode():
        res = reporter(cache_true[layer].to(device)).sigmoid()
    t_strs = [s.replace('Ġ', ' ') for s in tokenizer.convert_ids_to_tokens(tqa_formated_dataset_data[0])]
    display(cv.tokens.colored_tokens(t_strs, res[0]))

for dataset_name in ('dbpedia_14', 'ag_news', 'imdb'):
    layer=47
    reporter = torch.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
    #reporter = elk.training.Reporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
    pp(reporter)
    reporter.eval()
    print(f'Probe gpt2-xl trained on {dataset_name} for {layer}:')
    visualize(layer, reporter)
# %%
#with torch.inference_mode():
#    output = gpt2_xl.forward(
#        imdb_samples,
#        output_hidden_states=True, 
#        output_attentions=True,
#    )
#hidden_states = output['hidden_states']
#layers_num = len(output['hidden_states'])
#pp(f'{layers_num=} {heads_num=}')

# %%
# Visualize probe scores across layers per each head:
# TODO
# attentions = output['attentions']
# heads_num = output['attentions'][1].size(1)
# att_layers_num = len(attentions)
# 
# head_layer_score = torch.zeros((att_layers_num, heads_num))
# for layer in range(att_layers_num):
#     dataset_name = 'imdb'
#     reporter = elk.training.Reporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
#     reporter.eval()
#     scores = reporter(attentions[layer])
#     head_layer_score[scores[;]]    
# 
# # Plot the induction scores for each head in each layer
# imshow(
#     head_layer_score, 
#     labels={"x": "Head", "y": "Layer"}, 
#     title="Probe Score by Head", 
#     text_auto=".2f",
#     width=900, height=400
# )

# %%

# %%
# TransformerLens 
ht_model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-xl")
ht_model.eval()
tokenizer = ht_model.tokenizer
pp(ht_model)

is_lens = isinstance(ht_model, HookedTransformer)
layers = ht_model.cfg.n_layers
heads = ht_model.cfg.n_heads

# %%
# Calculates a probe score on IMDB dataset using TransformerLens. And visualizes score on each layer.
assert is_lens

sample = imdb_ds['train']['text'][156]
sample_false = f'{sample}\nDid the reviewer find this movie good or bad?\nGood'
sample_true = f'{sample}\nDid the reviewer find this movie good or bad?\n Bad'
with torch.inference_mode():
    _, cache_false = ht_model.run_with_cache(sample_false, remove_batch_dim=True)
    _, cache_true = ht_model.run_with_cache(sample_true, remove_batch_dim=True)

for layer in range(layers):
    probe = elk.training.Reporter.load(
        f'./data/gpt2-xl/imdb/reporters/layer_{layer}.pt', 
        map_location=device
    )
    pp(probe)
    act0 = cache_false['mlp_out', layer][-1].to(device)
    act1 = cache_true['mlp_out', layer][-1].to(device)
    p0 = probe(act0).item() # (act0 @ probe['probe.0.weight'].T + probe['probe.0.bias']).sigmoid().item()
    p1 = probe(act1).item() # (act1 @ probe['probe.0.weight'].T + probe['probe.0.bias']).sigmoid().item()
    confidence = 0.5*(p0 + (1-p1))
    pp(f'l {layer} {p0=} {p1=} {confidence=}')

#%%
# Visualize probe scores across layers per each head:
head_layer_score = torch.zeros((layers, heads))

def induction_score_hook(
    result: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
):
    global head_layer_score
    layer = hook.layer()
    probe = elk.training.Reporter.load(
        f'./data/gpt2-xl/imdb/festive-elion/reporters/layer_{layer}.pt', 
        map_location=device
    )
    assert result.size(0) == 1, f"Expecting one sample but: {result.shape}"
    score = probe(result[0, -1, :, :]).sigmoid()
    head_layer_score[layer, :] = score

ht_model.run_with_hooks(
    tokenizer.encode(sample_false, return_tensors='pt'),
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        lambda name: name.endswith("result"),
        induction_score_hook
    )]
)
# Plot the induction scores for each head in each layer
imshow(
    head_layer_score, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Probe Score by Head", 
    text_auto=".2f",
    width=900, height=400
)
