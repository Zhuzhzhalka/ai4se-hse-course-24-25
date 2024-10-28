import datasets
import pandas as pd
import re

from pathlib import Path

def remove_links(data):
    return re.sub(r"\b\w+:\/\/[^\/\s]+(?:\/.*)?", "", data)

def remove_common_contractions(data):
    table = {
        "n't": " not",
        "'re": " are",
        "'m":  " am",
        "'ve": " have",
    }

    for old, new in table.items():
        data = data.replace(old, new)

    return data

def prepare_data(data):
    data = remove_links(data)
    data = remove_common_contractions(data)
    data = re.sub(r"[&^#*]", "", data)

    return data

def prepare(raw_data: Path) -> datasets.Dataset:
    dataset = pd.read_excel(raw_data)
    dataset.dropna(inplace=True)
    dataset.insert(0, "prepared", dataset["message"].apply(prepare_data))
    dataset.drop_duplicates(inplace=True)

    return datasets.Dataset.from_pandas(dataset)


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
