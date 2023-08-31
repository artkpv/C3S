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
        for _ in (1,): # range(np_rand.integers(2, 5)):
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
def create_tokenized_tqa_dataset(tokenizer, tqa_dataset, np_rand):
    tqa_formated_dataset_data = []
    tqa_formated_dataset_labels = []
    prompt = '[CLS] {} To this last question, the following answer, "{}", is {}.'
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
    return (tqa_formated_dataset_data, tqa_formated_dataset_labels)


def create_multisentence_imdb_ds(np_rand, tokenizer, max_sentences_num=4):
    '''
      Creates IMDB based dataset with a sample with multiple sentences.
      Returns a list of form:  [( (left, right), (left_labels, right_labels) ) .. ]
      where 
      - left and right are contrast pairs, i.e. samples of tokenized true and false 
        sentences each (several sentences in a sample).
      - left and right labels are whether a sentence from left or right is true.
    '''
    imdb_ds = load_dataset('imdb')
    init_prompt = tokenizer.encode("The following movie reviews express what sentiment?", return_tensors='pt').squeeze_(0)
    qa_prompt = "\n{}\n{}\n"  # Question and answer prompt.
    sentiments = ['negative', 'positive']
    num = len(imdb_ds['test']['text'])
    to_switch = np_rand.integers(0, 2, (num,))
    dataset = list()
    current_num = 0
    for i, e in enumerate(imdb_ds['test']):
        if current_num == 0:
            dataset += [[
                [init_prompt, init_prompt],
                [list(), list()]
            ]]
            current_num = np_rand.integers(1, max_sentences_num+1)
        is_right = to_switch[i]
        for pos in (0, 1):
            qa = qa_prompt.format(e['text'], sentiments[e['label'] - pos - to_switch[i]])
            qa_ids = tokenizer.encode(qa, return_tensors='pt').squeeze_(0)
            dataset[-1][0][pos] = torch.concat(dataset[-1][0][pos], qa_ids)
            dataset[-1][1][pos] += [(len(dataset[-1][0][pos]), (pos - is_right)%2)]
        current_num -= 1
        
    return dataset