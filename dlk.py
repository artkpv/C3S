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
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from pprint import pp
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import einops
import elk 

import circuitsvis as cv
#from promptsource.templates import DatasetTemplates
from plotly_utils import imshow
from functools import cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)

seed = 42
np_rand = np.random.default_rng(seed=42)
#%%

# %%
# GPT2-XL from HF
# %%
gpt2_xl : GPT2Model = GPT2Model.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl.eval()
pp(gpt2_xl)

# %%
# TruthfulQA based many statements dataset.
#@cache
def get_tqa_dataset():
    truthfulqa = load_dataset('truthful_qa', 'generation')
    correct_statements = []
    incorrect_statements = []
    for e in truthfulqa['validation']:
        q = e['question']
        correct_statements += [(q, s) for s in e['correct_answers']]
        incorrect_statements += [(q, s) for s in e['incorrect_answers']]
    # tqa_dataset = [([statement, ..], [label, ..]), .. ]
    # Each label is either 0 (false) or 1 (true) or -1 (unknown).
    tqa_dataset = []
    while correct_statements or incorrect_statements:
        x = []
        y = []
        for _ in range(np_rand.integers(2, 5)):
            label =  np_rand.integers(0,2)
            statements = (incorrect_statements, correct_statements)[label]
            if statements:
                q, s = statements.pop()
                x += [(q, s)]
                y += [label]
        if x and y:
            tqa_dataset.append((x, y))
    return tqa_dataset
# %%
tqa_dataset = get_tqa_dataset()

#%%
tqa_formated_dataset_data = []
tqa_formated_dataset_labels = []
prompt = ' {} To this last question, the following answer, "{}", is {}.'
for x, y in tqa_dataset:
    x_t : torch.Tensor = None
    labels = []
    for (q, s), true_label in zip(x, y):
        label =  (true_label + np_rand.integers(0,2))%2
        ids = tokenizer.encode(
            prompt.format(q, s, ('false', 'true')[label]), 
            return_tensors='pt'
        )
        ids.squeeze_(0)
        x_t = ids if x_t is None else torch.concat((x_t, ids), -1)
        labels += [(x_t.size(0)-1, true_label)]
    tqa_formated_dataset_data += [x_t]
    tqa_formated_dataset_labels += [labels]
# %%
pp(tqa_formated_dataset_data[0].shape)
pp(tqa_formated_dataset_labels[0])
# %%
def visualize(ids, tokenizer, probs, labels):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    pp('. '.join([f'{tokens[pos-1]} {probs[pos-1]} - {l}' for pos, l in labels]))
    t_strs = [s.replace('Ġ', ' ') for s in tokens]
    display(cv.tokens.colored_tokens(t_strs, probs))
# %% 
# Calculate accuracy
layer=47
dataset_name = 'dbpedia_14'
reporter = elk.training.Reporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
reporter.eval()
correct_num = 0
all_num = 0
for data, labels in tqdm(list(zip(tqa_formated_dataset_data, tqa_formated_dataset_labels))):
    with torch.inference_mode():
        output = gpt2_xl.forward(
            data.to(device),
            output_hidden_states=True
        )
        res = reporter(output['hidden_states'][layer][0].to(device)).sigmoid()
        #visualize(data, tokenizer, res, labels)
        for (pos, l) in labels:
            all_num += 1
            if (res[pos-1] > 0.5) == (l == 1):
                correct_num += 1
    print(f'''Accuracy for TruthfulQA, using GPT2-xl, and probe from {dataset_name} on {layer}:
{correct_num}/{all_num} = {correct_num/all_num:.2}.''')
#%%
with torch.inference_mode():
    output = gpt2_xl.forward(
        tqa_formated_dataset_data[0],
        output_hidden_states=True, 
        output_attentions=True,
    )
hidden_states = output['hidden_states']
layers_num = len(output['hidden_states'])
heads_num = output['attentions'][1].size(1)
pp(f'{layers_num=} {heads_num=}')

# %% 
# Output shapes
pp(type(output))
pp(output.keys())
pp(output['last_hidden_state'].shape)
pp(len(output['hidden_states']))
pp(len(output['attentions']))
pp(len(output['past_key_values']))
pp('hidden states:')
for i,e in enumerate(hidden_states):
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
        res = reporter(hidden_states[layer].to(device)).sigmoid()
    t_strs = [s.replace('Ġ', ' ') for s in tokenizer.convert_ids_to_tokens(dataset[0][0])]
    display(cv.tokens.colored_tokens(t_strs, res[0]))

for dataset_name in ('dbpedia_14', 'ag_news', 'imdb'):
    layer=47
    reporter = elk.training.Reporter.load(f'./data/gpt2-xl/{dataset_name}/reporters/layer_{layer}.pt', map_location=device)
    reporter.eval()
    print(f'Probe gpt2-xl trained on {dataset_name} for {layer}:')
    visualize(layer, reporter)

# %% 
# IMDB dataset
# %%
imdb_ds = load_dataset('imdb')
imdb_samples = [
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
# %%
with torch.inference_mode():
    output = gpt2_xl.forward(
        imdb_samples,
        output_hidden_states=True, 
        output_attentions=True,
    )
hidden_states = output['hidden_states']
layers_num = len(output['hidden_states'])
pp(f'{layers_num=} {heads_num=}')

# %%
# Visualize probe scores across layers per each head:
attentions = output['attentions']
heads_num = output['attentions'][1].size(1)
att_layers_num = len(attentions)

head_layer_score = torch.zeros((att_layers_num, heads_num))
# TODO
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
# TransformerLens 
ht_model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-xl")
ht_model.eval()
tokenizer = ht_model.tokenizer
pp(ht_model)

is_lens = isinstance(ht_model, HookedTransformer)
layers = ht_model.cfg.n_layers
heads = ht_model.cfg.n_heads

# %%
assert is_lens

sample = imdb_ds['train']['text'][156]
sample_false = f'{sample}\nDid the reviewer find this movie good or bad?\nGood'
sample_true = f'{sample}\nDid the reviewer find this movie good or bad?\n Bad'
with torch.inference_mode():
    _, cache_false = ht_model.run_with_cache(sample_false, remove_batch_dim=True)
    _, hidden_states = ht_model.run_with_cache(sample_true, remove_batch_dim=True)

for layer in range(layers):
    probe = elk.training.Reporter.load(
        f'./data/gpt2-xl/imdb/festive-elion/reporters/layer_{layer}.pt', 
        map_location=device
    )
    pp(probe)
    act0 = cache_false['mlp_out', layer][-1].to(device)
    act1 = hidden_states['mlp_out', layer][-1].to(device)
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

# %%
# Experiments with T5 (UnifiedQA) model
# %%
def t5_experiments():

    model_name = "allenai/unifiedqa-t5-base" 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    sample = imdb_ds['train']['text'][156:157]
    input_ids = tokenizer.encode(f'''
        {sample}
        Did the reviewer find this movie good or bad?''', return_tensors="pt")
    # TODO: broken forward call.
    # output = model.forward(input_ids=input_ids, output_hidden_states=True)
    # pp(f"{output['decoder_hidden_states'][0].shape=}")
    # l11cp = elk.training.Reporter.load(f'./data/allenai/unifiedqa-t5-base/imdb/quirky-neumann/reporters/layer_11.pt', map_location=device)
    # pp((output['decoder_hidden_states'][0][0,-1] == output['decoder_hidden_states'][0][1,-1]).float().mean())
    # pp(l11cp(output['decoder_hidden_states'][0][0,-1]))
    # pp(l11cp(output['decoder_hidden_states'][0][1,-1]))

#t5_experiments()