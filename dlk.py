#%%

from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from pprint import pp
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

import elk 

import circuitsvis as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Experiments with T5 (UnifiedQA) model
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

#t5_experiments()

# %%
# Experiments with GPT2-XL
# Loading. Warning. This takes +16GB of RAM.
gpt2_xl: HookedTransformer = HookedTransformer.from_pretrained("gpt2-xl")
gpt2_xl.eval()
tokenizer = gpt2_xl.tokenizer
pp(gpt2_xl)

# %%
# Or can use it from HuggingFace:

# gpt2_xl : GPT2Model = GPT2Model.from_pretrained('gpt2-xl')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# gpt2_xl.eval()
# pp(gpt2_xl)

# %%
sample = imdb_ds['train']['text'][156]
sample_false = f'{sample}\nDid the reviewer find this movie good or bad?\nGood'
sample_true = f'{sample}\nDid the reviewer find this movie good or bad?\n Bad'
with torch.inference_mode():
    _, cache_false = gpt2_xl.run_with_cache(sample_false, remove_batch_dim=True)
    _, cache = gpt2_xl.run_with_cache(sample_true, remove_batch_dim=True)

#%%
for layer in range(1, 48):
    probe = torch.load(f'./data/gpt2-xl/imdb/festive-elion/reporters/layer_{layer}.pt')
    pp(probe.key())
    act0 = cache_false['mlp_out', layer][-1].cpu()
    act1 = cache['mlp_out', layer][-1].cpu()
    p0 = (act0 @ probe['probe.0.weight'].T + probe['probe.0.bias']).sigmoid().item()
    p1 = (act1 @ probe['probe.0.weight'].T + probe['probe.0.bias']).sigmoid().item()
    confidence = 0.5*(p0 + (1-p1))
    pp(f'l {layer} {p0=} {p1=} {confidence=}')


# %%
# Experimenting with TruthfulQA dataset.
# %%
reporter = elk.training.Reporter.load(f'./data/gpt2-xl/imdb/festive-elion/reporters/layer_47.pt', map_location=device)
pp(reporter)

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
#tokenizer = gpt2_xl.tokenizer
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
# with torch.inference_mode():
#     _, cache_true = gpt2_xl.run_with_cache(dataset[0][0])
# pp(cache_true['mlp_out', 47].shape)

with torch.inference_mode():
    output = gpt2_xl.forward(dataset[0][0], output_hidden_states=True)
    cache = output['hidden_states']
# %%
pp(f'{len(cache)=}')
pp(f'{cache[48].shape=}')
# %%
reporter = elk.training.Reporter.load(f'./data/gpt2-xl/dbpedia_14/reporters/layer_47.pt', map_location=device)
reporter.eval()
pp(reporter)
# %%
with torch.inference_mode():
    #res = reporter(cache_true['mlp_out', 47][0]).sigmoid()
    res = reporter(cache[47].to(device))[0].sigmoid()
pp(res.shape)
pp(dataset[0][1])
for inx, label in dataset[0][1]:
    print(inx, label)
    pp(res[inx-1])
# %%

reporter = elk.training.Reporter.load(f'./data/gpt2-xl/ag_news/reporters/layer_47.pt', map_location=device)
reporter.eval()
pp(reporter)

# %%
with torch.inference_mode():
    #res = reporter(cache_true['mlp_out', 47][0]).sigmoid()
    res = reporter(cache[47].to(device))[0].sigmoid()
pp(res.shape)
pp(dataset[0][1])
for inx, label in dataset[0][1]:
    print(inx, label)
    pp(res[inx-1])
# %%
reporter = torch.load(f'./data/gpt2-xl/dbpedia_14/lr_models/layer_47.pt', map_location=device)[0]
pp(reporter)
# %%
with torch.inference_mode():
    #res = reporter(cache_true['mlp_out', 47][0]).sigmoid()
    res = reporter(cache[47].to(device))
    res = res[0]
pp(res.shape)
pp(res)
pp(dataset[0][1])
for inx, label in dataset[0][1]:
    print(inx, label)
    pp(res[inx-1])
# %%

t_strs = [s.replace('Ġ', ' ') for s in tokenizer.convert_ids_to_tokens(dataset[0][0])]
pp(t_strs)

cv.tokens.colored_tokens(t_strs, res)
# %%
pp(tokenizer.decode(dataset[0][0]))