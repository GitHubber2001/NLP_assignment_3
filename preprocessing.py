import os

import datasets
import pandas as pd
from sklearn import model_selection
from transformers import AutoTokenizer, BatchEncoding


def merge_colums(dataframes: list):
    """Drops and merges the title and description columns into a text column"""

    for dataframe in dataframes:
        dataframe["text"] = dataframe["title"] + " " + dataframe["description"]
        dataframe.drop(columns=["title", "description"], inplace=True)


def preprocessing(random_seed: int):
    """Returns preprocessed train, validation and test sets from the dataset"""

    test_df = pd.read_json(os.path.join("data", "test.jsonl"), lines=True)
    train_df = pd.read_json(os.path.join("data", "train.jsonl"), lines=True)

    train_df, validation_df = model_selection.train_test_split(
        train_df, random_state=random_seed, test_size=0.1, train_size=0.9
    )

    train_df["label"] -= 1
    validation_df["label"] -= 1
    test_df["label"] -= 1

    merge_colums([train_df, validation_df, test_df])

    return (train_df, validation_df, test_df)


def setup_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)

    return tokenize


def tokenize_datasets(
    tokenize_function,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    new_train = datasets.Dataset.from_pandas(train)
    new_evaluate = datasets.Dataset.from_pandas(evaluate)
    new_test = datasets.Dataset.from_pandas(test)
    return (
        new_train.map(tokenize_function, batched=True),
        new_evaluate.map(tokenize_function, batched=True),
        new_test.map(tokenize_function, batched=True),
    )
