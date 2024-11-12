from collections.abc import Iterable
from functools import cache
from pprint import pprint

import datasets
import evaluate


@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))


def predict(dataset: datasets.Dataset, model: str) -> None:
    # Implement your function name prediction loop here
    predictions = ['func_one', 'func_three']
    references = ['func_one', 'func_two']

    eval_results = run_evaluate(predictions=predictions, references=references)
    print()
    print('*' * 80)
    print('Evaluation results:')
    pprint(eval_results)
    print('*' * 80)
    print()


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
