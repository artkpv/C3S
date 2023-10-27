# %%
from datasets import load_dataset
from jinja2 import Environment, PackageLoader, select_autoescape
import torch


# %%
def get_question_answer_dataset():
    """
    Returns dataset with prompts for single answer, for disjunction and for conjunction.
    """
    truthfulqa = load_dataset("truthful_qa", "generation")  # 817 rows
    env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())
    qa_t = env.get_template("question_answer.jinja")
    qas_t = env.get_template("question_answers.jinja")
    qas_l_t = env.get_template("question_answers_one_line.jinja")
    qa_dataset = []
    qas_dataset = []
    for row in truthfulqa["validation"]:
        qas_dataset.append(
            {
                "input": qas_t.render(
                    row,
                    a_A=row["incorrect_answers"][0],
                    a_B=row["correct_answers"][0],
                    is_disjunction=True,
                    label=True,
                ),
                "label": True,
                "is_correct": True,
            }
        )
        qa_dataset.append(
            {
                "input": qa_t.render(
                    row,
                    is_correct_answer=True,
                    label=True,
                ),
                "label": True,
                "is_correct": True,
            }
        )
    return qa_dataset, qas_dataset


#qa_dataset, qas_dataset = get_question_answer_dataset()
#print(qa_dataset)
#print(qas_dataset)


# %%
def get_tqa_dataset(np_rand):
    """

    Returns array of (x, y) pairs, where x is an array of (q, s),
    question and statement, and y is an array of true of false labels.
    """
    env = Environment(loader=PackageLoader("utils"), autoescape=select_autoescape())
    template = env.get_template("mytemplate.jinja")
    res = template.render(
        question="My super question",
        options=[("A", "First answer"), ("B", "Second answer")],
    )

    truthfulqa = load_dataset("truthful_qa", "generation")
    correct_statements = []
    incorrect_statements = []
    for e in truthfulqa["validation"]:
        q = e["question"]
        correct_statements += [(q, s) for s in e["correct_answers"]]
        incorrect_statements += [(q, s) for s in e["incorrect_answers"]]
    # tqa_dataset = [([statement, ..], [label, ..]), .. ]
    # Each label is either 0 (false) or 1 (true) or -1 (unknown).
    tqa_dataset = []
    while correct_statements or incorrect_statements:
        x = []
        y = []
        for _ in (1,):  # range(np_rand.integers(2, 5)):
            label = np_rand.integers(0, 2)
            statements = (incorrect_statements, correct_statements)[label]
            if statements:
                q, s = statements.pop()
                x += [(q, s)]
                y += [label]
        if x and y:
            tqa_dataset.append((x, y))
    return tqa_dataset


def create_tokenized_tqa_dataset(tokenizer, tqa_dataset, np_rand):
    tqa_formated_dataset_data = []
    tqa_formated_dataset_labels = []
    prompt = '[CLS] {} To this last question, the following answer, "{}", is {}.'
    for x, y in tqa_dataset:
        x_t: torch.Tensor = None
        labels = []
        for (q, s), true_label in zip(x, y):
            label = (true_label + np_rand.integers(0, 2)) % 2
            ids = tokenizer.encode(
                prompt.format(q, s, ("false", "true")[label]), return_tensors="pt"
            )
            ids.squeeze_(0)
            x_t = ids if x_t is None else torch.concat((x_t, ids), -1)
            labels += [(x_t.size(0) - 1, true_label)]
        tqa_formated_dataset_data += [x_t]
        tqa_formated_dataset_labels += [labels]
    return (tqa_formated_dataset_data, tqa_formated_dataset_labels)
