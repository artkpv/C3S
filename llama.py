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

# %%

qa_dataset, qas_dataset = get_question_answer_dataset()
pp(qas_dataset[0])
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

#%%
t_output = tokenizer(qa_dataset[0]['input'], return_tensors="pt")
pp(t_output)


#%%
outputs = model(**t_output, output_hidden_states=True)
# %%
pp(outputs.logits.shape)
pp(tokenizer.decode(outputs.logits[:, -1]))

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
