import gzip
import json
import requests
from typing import List, Optional


def download_c4_dataset(url: str, output_path: str) -> None:
    with open(output_path, "wb") as f:
        f.write(requests.get(url).content)


def load_c4_data(file_path: str) -> List[dict]:
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def truncate_text_at_word(text: str, max_chars: int = 300) -> str:
    words = text.split()
    current_text = ""

    for word in words:
        if len(current_text) + len(word) + 1 > max_chars:
            break
        if current_text == "":
            current_text = word
        else:
            current_text += " " + word

    return current_text


def prepare_prompts(data: List[dict], max_chars: int = 300, max_count: Optional[int] = None) -> List[str]:
    prompts = [
        truncate_text_at_word(example["text"], max_chars=max_chars)
        for example in data
    ]

    if max_count is not None:
        prompts = prompts[:max_count]

    return prompts


def setup_c4_dataset(dataset_url: str = None, dataset_path: str = "c4-validation.00000-of-00001.json.gz") -> List[str]:

    # Download if URL provided
    if dataset_url:
        download_c4_dataset(dataset_url, dataset_path)

    # Load and prepare data
    data = load_c4_data(dataset_path)
    prompts = prepare_prompts(data)

    return prompts