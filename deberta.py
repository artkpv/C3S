# Deberta-v2-xxlarge model
#%%
from tqdm import tqdm
import numpy as np
import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer
from datasets import load_dataset
from pprint import pp

#from promptsource.templates import DatasetTemplates
from utils.datasets import get_tqa_dataset, create_multisentence_imdb_ds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp(device)

seed = 42
np_rand = np.random.default_rng(seed=42)



# %%
# Deberta
model : DebertaV2Model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xxlarge-mnli')
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge-mnli')
model.eval()
pp(model)

# %%
# Create dataset

def create_multisentence_imdb_ds(np_rand, max_sentences_num=4):
    '''
      Creates IMDB based dataset with a sample with multiple sentences.
      Returns a list of form:  [( (left, right), (left_labels, right_labels) ) .. ]
      where 
      - left and right are contrast pairs, i.e. samples of tokenized true and false 
        sentences each (several sentences in a sample).
      - left and right labels are whether a sentence from left or right is true.
    '''
    imdb_ds = load_dataset('imdb', split='test[:10%]')
    init_prompt = "The following movie reviews express what sentiment?"
    qa_prompt = "\n{}\n{}\n"  # Question and answer prompt.
    sentiments = ['negative', 'positive']
    num = len(imdb_ds['text'])
    to_switch = np_rand.integers(0, 2, (num,))
    dataset = list()
    current_num = 0
    for i, e in enumerate(imdb_ds):
        if current_num == 0:
            dataset += [[
                [init_prompt, init_prompt],
                [list(), list()]
            ]]
            current_num = np_rand.integers(1, max_sentences_num+1)
        is_right = to_switch[i]
        for pos in (0, 1):
            qa = qa_prompt.format(e['text'], sentiments[e['label'] - pos - to_switch[i]])
            dataset[-1][0][pos] = torch.concat(dataset[-1][0][pos], qa)
            dataset[-1][1][pos] += [(len(dataset[-1][0][pos]), (pos - is_right)%2)]
        current_num -= 1
        
    return dataset
ds = create_multisentence_imdb_ds(np_rand)

# %%
ds

# %%
tqa_dataset = get_tqa_dataset(np_rand)
tqa_formated_dataset_data : list, tqa_formated_dataset_labels : list = create_tokenized_tqa_dataset(
    deberta_tokenizer, tqa_dataset, np_rand)

# %% 
# Calculate accuracy
layer=47
dataset_name = 'dbpedia_14'
reporter = torch.load(f'./data/deberta-v2-xxlarge-mnli/imdb/eager-colden/reporters/layer_{layer}.pt', map_location=device)
#reporter.eval()
correct_num = 0
all_num = 0
set_num = len(tqa_formated_dataset_data)
SOME_SUBSET=30
selected_indeces = torch.randperm(set_num)[:SOME_SUBSET]
subset = [e for i,e in enumerate(zip(tqa_formated_dataset_data, tqa_formated_dataset_labels)) if i in selected_indeces]
for data, labels in tqdm(subset):
    data = data.unsqueeze(0)
    pp(data.shape)
    with torch.inference_mode():
        output = model.forward(
            data.to(device),
            output_hidden_states=True
        )
        h_states = output['hidden_states'][layer][0].to(device)
        logits = reporter(h_states)
        pp(f'{labels=}')
        res = logits.sigmoid()
        pp(f'{logits[[0,-1]]=}')
        pp(f'{res[[0,-1]]=}')
        #visualize(data, tokenizer, res, labels)
        for (pos, l) in labels:
            all_num += 1
            if (res[0] > 0.5) == (l == 1):
                correct_num += 1
            #if (res[pos-1] > 0.5) == (l == 1):
            #    correct_num += 1
    print(f'{correct_num}/{all_num}')
print(f'''Accuracy for TruthfulQA, using GPT2-xl, and probe from {dataset_name} on {layer}: {correct_num}/{all_num} = {correct_num/all_num:.1}.''')

# %% 
# IMDB dataset
imdb_ds = load_dataset('imdb')
imdb_samples = [
f'''
{imdb_ds['train']['text'][:1]}
Did the reviewer find this movie good or bad? 
{a}
''' for a in ('good', 'bad')
]

