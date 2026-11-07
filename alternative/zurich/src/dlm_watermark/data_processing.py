from datasets import Dataset
from transformers import AutoTokenizer
from typing import Callable

def tokenize_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, user_prompt: str, length: int = 50
):
    def tokenize_function(
        example, length: int = length, tokenizer: AutoTokenizer = tokenizer
    ):
        return tokenizer(
            user_prompt.format(**example),
            max_length=length,
            truncation=True
        )

    dataset = dataset.map(tokenize_function)

    return dataset

def tokenize_dataset_with_chat(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 2048, pad: bool = False,
):
    def tokenize_function(
        example, max_length: int = max_length, tokenizer: AutoTokenizer = tokenizer
    ):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            padding="max_length" if pad else False,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=True,
        )

    dataset = dataset.map(tokenize_function)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def convert_sft_dataset(
    ds: Dataset,
    convert_fn: Callable,
):

    ds = ds.map(convert_fn)
    return ds
