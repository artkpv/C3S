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

from plotly_utils import imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)


# %%
# GPT2-XL from HF
# %%
gpt2_xl : GPT2Model = GPT2Model.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl.eval()
pp(gpt2_xl)

# %%
# Experimenting with TruthfulQA dataset.

# %%
truthfulqa = load_dataset('truthful_qa', 'generation')
# Construct statements from each correct_answer and incorrect_answer:
correct_statements = []
incorrect_statements = []
for e in truthfulqa['validation']:
    for correct_answer in e['correct_answers']:
        correct_statements.append(f"{e['question']} {correct_answer}.")
    for incorrect_answer in e['incorrect_answers']:
        incorrect_statements.append(f"{e['question']} {incorrect_answer}.")
pp(len(incorrect_statements))
pp(len(correct_statements))

# %%
# Create dataset with x as concatenated correct and incorrect 2..4 statements,
# and y as several 0 or 1 depending on whether a correct or incorrect statement is the correct answer.
dataset = []
while correct_statements or incorrect_statements:
    x : torch.Tensor = None
    y = []
    for _ in range(np.random.randint(2, 5)):
        label =  np.random.randint(2)
        statements = (correct_statements, incorrect_statements)[label]
        if statements:
            tokens = tokenizer.encode( statements.pop(), return_tensors='pt')
            x = tokens if x is None else torch.concat((x, tokens), -1)
            inx = tokens.shape[1] + (y[-1][0] if y else 0)
            y.append((inx, label))
    if x is not None:
        x.squeeze_(0)
        dataset.append((x, y))
pp(dataset[0])        

# %%
with torch.inference_mode():
    output = gpt2_xl.forward(
        dataset[0][0],
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