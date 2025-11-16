from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.models.model_factory import load_model
import yaml
import argparse
import glob
import pandas as pd
import os
from markllm.evaluation.tools.text_editor import (
    WordDeletion,
    SynonymSubstitution,
    ContextAwareSynonymSubstitution,
    DipperParaphraser,
    BackTranslationTextEditor,
    TextEditor
)
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BertTokenizer,
    BertForMaskedLM,
)
from tqdm import tqdm
import nltk
from typing import Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor  # Added for parallelization
 

client = OpenAI(api_key=os.getenv("TRANSLATION_API_KEY"))

TRANSLATION_API_KEY = os.getenv("TRANSLATION_API_KEY")

nltk.download("punkt_tab")

def parallel_paraphrase(paraphraser, texts, max_workers=128, desc="Paraphrasing"):
    """Paraphrase a list of texts in parallel.

    Parameters:
        paraphraser: An object with an `.edit(text)` method.
        texts (List[str]): The texts to paraphrase.
        max_workers (int): Number of parallel worker threads.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List[str]: The paraphrased texts, in the original order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # `executor.map` preserves the input order, so output order matches `texts`.
        return list(tqdm(executor.map(paraphraser.edit, texts), total=len(texts), desc=desc))


def parallel_edit(editor, texts, prompts, max_workers=128, desc="Editing"):
    """Apply an editor's `.edit(text, prompt)` over texts in parallel.

    Parameters:
        editor: An object with an `.edit(text, prompt)` method.
        texts (List[str]): The texts to edit.
        prompts (List[str]): The corresponding prompts/references.
        max_workers (int): Number of parallel worker threads.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List[str]: The edited texts, in the original order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def _call(pair):
            text, prompt = pair
            return editor.edit(text, prompt)

        return list(
            tqdm(
                executor.map(_call, zip(texts, prompts)),
                total=len(texts),
                desc=desc,
            )
        )



class OriginalTextEditor:
    def edit(self, text, *args, **kwargs):
        return text
    
class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.prompt = prompt

        client = OpenAI()
        self.client = client

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""

        response = self.client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input= self.prompt + text,
        )
        return response.output_text

class OpenAITranslatorProvider:
    """
    class that wraps functions, which use the ChatGPT
    under the hood to translate word(s)
    """

    def __init__(
        self,
        source: str = "auto",
        target: str = "english",
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-5-nano-2025-08-07",
        **kwargs,
    ):
        """
        @param api_key: your openai api key.
        @param source: source language
        @param target: target language
        """

        self.api_key = api_key
        self.model = model

        self.source = source
        self.target = target

    def translate(self, text: str, **kwargs) -> str:
        """
        @param text: text to translate
        @return: translated text
        """


        prompt = f"Translate the text below into {self.target}.\n"
        prompt += f'Text: "{text}"'

        response = client.chat.completions.create(model=self.model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ])

        translated_text = response.choices[0].message.content

        return translated_text

DELETIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
SUBSTITUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

IGNORED_COLUMNS = ["prompt", "completion", "ages","z_score", "token_color", "mask", "ppl", "p_value"]

def parse_args():
    parser = argparse.ArgumentParser(description="Ablate generation parameters")
    parser.add_argument("--path", type=str, help="Path to the data to evaluate")
    parser.add_argument("--paths", type=str, help="Path to the folder to evaluate")
    parser.add_argument("--config", type=str, help="Path to the config")
    parser.add_argument(
        "--original", action="store_true", help="Evaluate deletion attacks"
    )
    parser.add_argument(
        "--deletion", action="store_true", help="Evaluate deletion attacks"
    )
    parser.add_argument(
        "--substitution", action="store_true", help="Evaluate substitution attacks"
    )
    parser.add_argument(
        "--ca_substitution",
        action="store_true",
        help="Evaluate context-aware substitution attacks",
    )
    parser.add_argument(
        "--gpt_paraphraser",
        action="store_true",
        help="Evaluate GPT paraphraser attacks",
    )
    parser.add_argument(
        "--dipper_paraphraser",
        action="store_true",
        help="Evaluate Dipper paraphraser attacks",
    )
    parser.add_argument(
        "--translation",
        action="store_true",
        help="Evaluate back translation attacks",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    default_config = args.config
    config = MainConfiguration(**yaml.safe_load(open(default_config, "r")))

    if args.overwrite:
        config.evaluation_config.skip_if_exists = False

    if args.paths:
        paths = glob.glob(args.paths + "/**/*.jsonl", recursive=True)
    else:
        paths = [args.path]

    print(paths)

    # Getting attacks
    attacks = {}
    if args.original:
        original = [
            (
                OriginalTextEditor(),  # No attack for original
                {"parameter": None},
            )
        ]
        attacks["original"] = original
    if args.deletion:
        deletions = [
            (
                WordDeletion(ratio=deletion),
                {"parameter": deletion},
            )
            for deletion in DELETIONS
        ]
        attacks["deletion"] = deletions
    if args.substitution:
        substitutions = [
            (
                SynonymSubstitution(ratio=substitution),
                {"parameter": substitution},
            )
            for substitution in SUBSTITUTIONS
        ]
        attacks["substitution"] = substitutions
    if args.ca_substitution:
        ca_substitutions = [
            (
                ContextAwareSynonymSubstitution(
                    ratio=substitution,
                    tokenizer=BertTokenizer.from_pretrained("bert-large-uncased"),
                    model=BertForMaskedLM.from_pretrained("bert-large-uncased", device_map="auto"),
                ),
                {"parameter": substitution},
            )
            for substitution in SUBSTITUTIONS
        ]
        attacks["ca-substitution"] = ca_substitutions
    if args.gpt_paraphraser:
        gpt_paraphrasers = [
            (
                GPTParaphraser(
                    prompt="Please rewrite the following text: ",
                ),
                {"parameter": None},
            )
        ]
        attacks["gpt-paraphraser"] = gpt_paraphrasers

    if args.dipper_paraphraser:
        dipper_paraphrasers = [
            (
                DipperParaphraser(
                    tokenizer=T5Tokenizer.from_pretrained(
                        "google/t5-v1_1-xxl"
                    ),
                    model=T5ForConditionalGeneration.from_pretrained(
                        "kalpeshk2011/dipper-paraphraser-xxl",
                        device_map="auto",
                    ),
                    lex_diversity=60,
                    order_diversity=0,
                    sent_interval=1,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=0.75,
                    top_k=None,
                ),
                {"parameter": None},
            )
        ]
        attacks["dipper-paraphraser"] = dipper_paraphrasers

    if args.translation:
        translators = [
            (
                BackTranslationTextEditor(
                    translate_to_intermediary=OpenAITranslatorProvider(
                        source="en", target="zh", api_key=TRANSLATION_API_KEY
                    ).translate,
                    translate_to_source=OpenAITranslatorProvider(
                        source="zh", target="en", api_key=TRANSLATION_API_KEY
                    ).translate,
                ),
                {"parameter": None},
            )
        ]
        attacks["back-translation"] = translators


    n_iteration = len(attacks) * len(paths)
    print(f"Total number of iterations: {n_iteration}")

    for path in paths:
        print(f"Evaluating {path}")
        config.evaluation_config.save_path = path
        exetensionless_path = path.split(".")[:-1]
        exetensionless_path = ".".join(exetensionless_path)
        print(exetensionless_path)

        print(f"Evaluating watermark detection on {path}")
        _, tokenizer = load_model(
            model_config=config.model_configuration, tokenizer_only=True
        )
        watermark = load_watermark_from_config(
            config=config.watermark_config,
            tokenizer=tokenizer,
            watermark_type=config.watermark_type,
        )
        device = watermark.device

        df = pd.read_json(path, lines=True)
        completions = df["completion"].tolist()
        prompts = df["prompt"].tolist()

        # Put all the non-ignored columns in a list of dictionaries
        info_columns = [
            {col: str(df.loc[i, col]) for col in df.columns if col not in IGNORED_COLUMNS} for i in range(len(df))
        ]

        for attack_name, attack_list in attacks.items():
            attack_res = []
            save_path = f"{exetensionless_path}_{attack_name}.csv"

            if not args.overwrite and os.path.exists(save_path):
                print(f"Results for {attack_name} already exist at {save_path}. Skipping.")
                continue

            for attack, params in attack_list:
                print(f"Evaluating {attack_name} with parameters {params} on {path}")

                lines = []

                # Parallelize paraphrasing and back-translation; keep others sequential
                if attack_name in {"gpt-paraphraser", "dipper-paraphraser", "back-translation"}:
                    # Run the editing in parallel while preserving order
                    edited_texts = parallel_edit(
                        attack,
                        completions,
                        prompts,
                        max_workers=128,
                        desc=f"{attack_name} (parallel)",
                    )

                    for edited_completion, prompt, inf in tqdm(
                        zip(edited_texts, prompts, info_columns), total=len(edited_texts)
                    ):
                        inputs = tokenizer(edited_completion, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        convolution_kernel = eval(inf["convolution_kernel"])
                        watermark.update_conv_kernel(convolution_kernel)

                        detection_output = watermark.detect(
                            **inputs,
                        )
                        if "p_value" not in detection_output:
                            print(
                                f"Warning: 'p_value' not found in detection output for {attack_name} on {path}."
                            )
                            pvalue = None
                        else:
                            pvalue = detection_output["p_value"]
                            lines.append(
                                {
                                    **inf,  # Unpack the info columns
                                    "p_value": pvalue,
                                    "attack_name": attack_name,
                                    "attack_params": params,
                                }
                            )

                        if args.verbose:
                            print(
                                f"Attack: {attack_name}, Parameters: {params}, "
                                f"Prompt: {prompt}, "
                                f"Edited Completion: {edited_completion}, "
                                f"P-value: {pvalue}"
                            )
                else:
                    # Sequential path for other attacks
                    for completion, prompt, inf in tqdm(
                        zip(completions, prompts, info_columns), total=len(completions)
                    ):
                        edited_completion = attack.edit(completion, prompt)
                        inputs = tokenizer(edited_completion, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        convolution_kernel = eval(inf["convolution_kernel"])
                        watermark.update_conv_kernel(convolution_kernel)

                        detection_output = watermark.detect(
                            **inputs,
                        )
                        if "p_value" not in detection_output:
                            print(
                                f"Warning: 'p_value' not found in detection output for {attack_name} on {path}."
                            )
                            pvalue = None
                        else:
                            pvalue = detection_output["p_value"]
                            lines.append(
                                {
                                    **inf,  # Unpack the info columns
                                    "p_value": pvalue,
                                    "attack_name": attack_name,
                                    "attack_params": params,
                                }
                            )

                        if args.verbose:
                            print(
                                f"Attack: {attack_name}, Parameters: {params}, "
                                f"Prompt: {prompt}, Completion: {completion}, "
                                f"Edited Completion: {edited_completion}, "
                                f"P-value: {pvalue}"
                            )
                attack_res.extend(lines)


            print(f"Saving results for {attack_name} on {save_path}")
            attack_df = pd.DataFrame(attack_res)
            attack_df.to_csv(
                save_path,
                index=False,
            )


if __name__ == "__main__":
    main()
