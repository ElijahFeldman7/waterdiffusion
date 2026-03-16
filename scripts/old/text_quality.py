import re
from typing import Dict

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_perplexity(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
    if not text.strip():
        return float("nan")
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    if input_ids.size(1) > 1024:
        input_ids = input_ids[:, :1024]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
    perplexity = torch.exp(neg_log_likelihood)
    return perplexity.item()


def count_syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_char_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_char_was_vowel:
            syllables += 1
        prev_char_was_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return syllables if syllables > 0 else 1


def evaluate_text_quality(text: str) -> Dict[str, float]:
    text = text.strip()
    if not text:
        return {
            "avg_sentence_length": float("nan"),
            "avg_word_length": float("nan"),
            "lexical_diversity": float("nan"),
            "flesch_reading_ease": float("nan"),
            "sequence_length": float("nan"),
        }
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b\w+\b", text.lower())
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    total_chars = sum(len(word) for word in words)
    total_syllables = sum(count_syllables(word) for word in words)
    unique_tokens = len(set(words))

    avg_sentence_length = len(words) / num_sentences
    avg_word_length = total_chars / num_words
    lexical_diversity = unique_tokens / num_words
    avg_syllables = total_syllables / num_words
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables

    return {
        "avg_sentence_length": float(avg_sentence_length),
        "avg_word_length": float(avg_word_length),
        "lexical_diversity": float(lexical_diversity),
        "flesch_reading_ease": float(flesch_reading_ease),
        "sequence_length": float(len(words)),
    }
