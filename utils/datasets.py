'''
Datasets
'''
from datasets import load_dataset
import torch

# Many statements dataset based on TruthfulQA
#@cache
def get_tqa_dataset(np_rand):
    '''
    Returns array of (x, y) pairs, where x is an array of (q, s), 
    question and statement, and y is an array of true of false labels.
    '''
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

#%%
def _create_tokenized_tqa_dataset(tokenizer, tqa_dataset, np_rand):
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
    #pp(tqa_formated_dataset_data[0].shape)
    #pp(tqa_formated_dataset_labels[0])
    return (tqa_formated_dataset_data, tqa_formated_dataset_labels)