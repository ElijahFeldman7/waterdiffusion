from dlm_watermark.configs import MainConfiguration
from dlm_watermark.models.model_factory import load_model
import yaml
import argparse
import os
from datasets import load_dataset, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Ablate generation parameters")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--name", type=str, help="Name of the generated dataset")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, help="Path of the dataset", default="allenai/c4")
    parser.add_argument("--dataset_name", type=str, help="Sub-Name of the dataset", default="realnewslike")
    parser.add_argument("--split", type=str, help="Dataset split to use", default="train")

    # Generated dataset argument
    parser.add_argument("--n_rows", type=int, default=600, help="Number of rows to process")
    parser.add_argument("--n_masked", type=int, default=200, help="Number of masked tokens to infill")
    parser.add_argument("--prefix_size", type=int, default=100, help="Size of the prefix")
    parser.add_argument("--suffix_size", type=int, default=100, help="Size of the suffix")
    args = parser.parse_args()

    return args

def process_infilling_prompt(prefix, suffix, tokenizer, number_of_mask):
    middle = [tokenizer.mask_token_id] * number_of_mask
    return tokenizer.decode(prefix + middle + suffix)

def main(args):

    config = MainConfiguration(**yaml.safe_load(open(args.config, "r")))
    _, tokenizer = load_model(config.model_configuration, tokenizer_only=True)

    dataset = load_dataset(args.dataset, name=args.dataset_name, split=args.split)
    infilling_dataset = []

    for i, row in enumerate(dataset):
        text = row["text"]
        encoded_text = tokenizer.encode(text)
        prefix = encoded_text[:args.prefix_size]
        suffix = encoded_text[args.prefix_size + args.n_masked:args.prefix_size + args.n_masked + args.suffix_size]
        infilling_dataset.append(process_infilling_prompt(prefix, suffix, tokenizer, args.n_masked))

        if i == args.n_rows-1:
            break

    infilling_dataset = Dataset.from_dict({"text": infilling_dataset})
    output_path = f"data/infilling/{args.name}.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    infilling_dataset.to_json(output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)