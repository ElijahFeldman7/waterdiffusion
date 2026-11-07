#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
import os
import sys
from typing import List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import check_watermark_compliance


class RobustWatermarkDetector:
    """Robust watermark detector using sliding window analysis."""

    def __init__(self, model_name: str = 'GSAI-ML/LLaDA-1.5', window_size: int = 16, private_key: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.window_size = window_size
        self.private_key = private_key

    def detect_z(self, match_values: List[float], gamma: float = 0.5, threshold: float = 1.25) -> Tuple[bool, float]:

        match_values = np.array(match_values)
        var = gamma * (1 - gamma) / self.window_size
        z_scores = (match_values - gamma) / np.sqrt(var)
        agg_z = np.mean(z_scores**2)
        is_detected = agg_z > threshold
        return is_detected, agg_z

    def calculate_window_ratios(self, prompt: str, generated_text: str,
                               min_tokens: int = 200) -> Optional[List[float]]:

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        prompt_len = len(prompt_ids)

        if len(generated_ids) < min_tokens:
            return None

        if len(generated_ids) < self.window_size:
            return None

        match_ratios = []

        # Sliding window analysis
        for j in range(len(generated_ids) - self.window_size + 1):
            window_ids = generated_ids[j:j + self.window_size]
            matched_count = 0

            for offset, token_id in enumerate(window_ids):
                real_pos = prompt_len + j + offset + 1  
                if check_watermark_compliance(real_pos, token_id, self.private_key):
                    matched_count += 1

            match_ratio = matched_count / self.window_size
            match_ratios.append(match_ratio)

        return match_ratios

    def analyze_csv_file(self, csv_path: str, prompt_col: int = 0, text_col: int = 1, threshold: float = 1.5,
                        min_tokens: int = 200) -> List[float]:

        try:
            try:
                df = pd.read_csv(csv_path)
                if len(df.columns) < max(prompt_col + 1, text_col + 1):
                    df = pd.read_csv(csv_path, header=None)
            except:
                df = pd.read_csv(csv_path, header=None)
        except FileNotFoundError:
            print(f"Error: File {csv_path} not found")
            return []

        z_scores = []

        df['token_count'] = df.iloc[:, text_col].apply(
            lambda x: len(self.tokenizer.encode(str(x), add_special_tokens=False)) if pd.notna(x) else 0
        )

        target_indices = df.index[df['token_count'] >= min_tokens]

        print(f"Processing {len(target_indices)} texts from {csv_path} (≥{min_tokens} tokens)")

        for i, target_row_index in enumerate(target_indices):
            row = df.iloc[target_row_index]
            prompt = str(row.iloc[prompt_col])
            generated_text = str(row.iloc[text_col])

            match_ratios = self.calculate_window_ratios(prompt, generated_text, min_tokens)

            if match_ratios is not None:
                is_detected, z_score = self.detect_z(match_ratios, threshold=threshold)
                z_scores.append(z_score)
            else:
                z_scores.append(np.nan)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(target_indices)} texts")

        z_scores = [z for z in z_scores if not np.isnan(z)]
        print(f"  Valid z-scores: {len(z_scores)}")

        return z_scores

    def save_z_scores(self, z_scores: List[float], output_path: str):
        """Save z-scores to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for score in z_scores:
                f.write(f"{score}\n")
        print(f"Z-scores saved to: {output_path}")

    def load_z_scores(self, file_path: str) -> List[float]:
        """Load z-scores from file."""
        try:
            return np.loadtxt(file_path).tolist()
        except:
            print(f"Error loading z-scores from {file_path}")
            return []

    def compute_auc(self, negative_scores_file: str, positive_scores_file: str) -> Optional[float]:

        scores_neg = self.load_z_scores(negative_scores_file)
        scores_pos = self.load_z_scores(positive_scores_file)

        if not scores_neg:
            print(f"Error: Could not load negative scores from {negative_scores_file}")
            return None

        if not scores_pos:
            print(f"Error: Could not load positive scores from {positive_scores_file}")
            return None

        y_true = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
        y_scores = np.concatenate([scores_neg, scores_pos])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc


def main():
    parser = argparse.ArgumentParser(description="Robust watermark detection with sliding window analysis")

    parser.add_argument('--mode', choices=['analyze', 'auc'], required=True,
                       help='Mode: analyze CSV files, plot ROC curves, or compute AUC')

    # Analysis mode arguments
    parser.add_argument('--input_csv', help='Input CSV file for analysis')
    parser.add_argument('--output_scores', help='Output file for z-scores')
    parser.add_argument('--prompt_col', type=int, default=0)
    parser.add_argument('--text_col', type=int, default=1)

    # Detection parameters
    parser.add_argument('--window_size', type=int, default=8, help='Sliding window size')
    parser.add_argument('--min_tokens', type=int, default=200, help='Minimum tokens for analysis')
    parser.add_argument('--model_name', default='GSAI-ML/LLaDA-1.5', help='Tokenizer model name')
    parser.add_argument('--private_key', help='Private key for watermark detection (must match key used for generation)')

    # Plotting mode arguments
    parser.add_argument('--original', help='File with non-watermarked z-scores')
    parser.add_argument('--watermark', nargs='+', help='Files with watermarked z-scores')

    args = parser.parse_args()

    # Initialize detector
    detector = RobustWatermarkDetector(args.model_name, args.window_size, args.private_key)

    if args.mode == 'analyze':
        if not args.input_csv or not args.output_scores:
            print("Error: --input_csv and --output_scores are required for analyze mode")
            sys.exit(1)

        print(f"Analyzing: {args.input_csv}")
        print(f"Window size: {args.window_size}")
        print(f"Minimum tokens: {args.min_tokens}")
        print(f"Private key: {'***provided***' if args.private_key else 'None (using position-based detection)'}")


        z_scores = detector.analyze_csv_file(
            args.input_csv,
            args.prompt_col,
            args.text_col,
            args.min_tokens
        )

        detector.save_z_scores(z_scores, args.output_scores)

        print(f"Total z-scores: {len(z_scores)}")
        if z_scores:
            print(f"Mean z-score: {np.mean(z_scores):.4f}")
            print(f"Std z-score: {np.std(z_scores):.4f}")


    elif args.mode == 'auc':
        if not args.original or not args.watermark:
            sys.exit(1)

        if len(args.watermark) != 1:
            print("Error: AUC mode requires exactly one positive score file")
            sys.exit(1)

        positive_file = args.watermark[0]
        print(f"Computing AUC:")

        auc_value = detector.compute_auc(args.original, positive_file)
        if auc_value is not None:
            print(f"\nAUC = {auc_value:.4f}")
        else:
            print("\nFailed to compute AUC")
            sys.exit(1)


if __name__ == "__main__":
    main()