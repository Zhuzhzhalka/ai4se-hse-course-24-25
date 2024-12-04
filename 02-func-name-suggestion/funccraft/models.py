import datasets
import evaluate
import torch
from collections.abc import Iterable
from functools import cache
from pprint import pprint
from transformers import T5ForConditionalGeneration, AutoTokenizer

device = torch.device("cpu")


@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))


def pretty_print(title, data):
    print()
    print('*' * 80)
    print(title)
    pprint(data)
    print('*' * 80)
    print()


def predict_function_name(tokenizer, model, function_body):
    inputs = tokenizer.encode(function_body, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict(dataset: datasets.Dataset, model: str) -> None:
    data = datasets.Dataset.to_pandas(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model).to(device)

    references = data['custom_fn_name']

    predictions = [
        predict_function_name(tokenizer, model, i).split(' ')
        for i in data['custom_with_comments']
    ]
    predictions = [i[1] if len(i) > 1 else i[0] for i in predictions]

    eval_results = run_evaluate(predictions, references)
    pretty_print(
        "Evaluation results for functions with comments:", eval_results
    )

    predictions = [
        predict_function_name(tokenizer, model, i).split(' ')
        for i in data['custom_without_comments']
    ]
    predictions = [i[1] if len(i) > 1 else i[0] for i in predictions]

    eval_results = run_evaluate(predictions, references)
    pretty_print(
        "Evaluation results for functions without comments:", eval_results
    )


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
