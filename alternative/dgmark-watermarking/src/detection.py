import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Tuple, Optional
import csv
import os
from utils import check_watermark_compliance


class WatermarkDetector:
    """Statistical watermark detection and analysis."""

    def __init__(self, eot_token_id: int = 126081, private_key: Optional[str] = None):
        self.eot_token_id = eot_token_id
        self.private_key = private_key

    def calculate_match_ratio(self, generated_ids: List[int], prompt_len: int) -> Tuple[int, int, float]:

        try:
            cutoff_index = generated_ids.index(self.eot_token_id)
            trimmed_ids = generated_ids[:cutoff_index + 1]
        except ValueError:
            trimmed_ids = generated_ids

        # Calculate watermark matches using key-based compliance
        matched_count = 0
        for j, token_id in enumerate(trimmed_ids):
            real_pos = prompt_len + j + 1  
            if check_watermark_compliance(real_pos, token_id, self.private_key):
                matched_count += 1

        match_ratio = matched_count / len(trimmed_ids) if trimmed_ids else 0.0
        return matched_count, len(trimmed_ids), match_ratio

    def load_match_ratios(self, file_path: str) -> List[float]:
        """Load match ratios from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [float(line.strip()) for line in f.readlines()]

    def compute_detection_stats(self, watermarked_ratios: List[float], original_ratios: List[float],
                               threshold_z: float = 5.0) -> dict:

        data1 = [ratio * 100 for ratio in watermarked_ratios]
        data0 = [ratio * 100 for ratio in original_ratios]

        mu0 = np.mean(data0)
        sigma0 = np.std(data0)

        z_data0 = [(x - mu0) / sigma0 for x in data0]
        z_data1 = [(x - mu0) / sigma0 for x in data1]

        z_dt0 = np.array(z_data0)
        z_dt1 = np.array(z_data1)

        fpr = np.mean(z_dt0 > threshold_z)  # False Positive Rate
        fnr = np.mean(z_dt1 <= threshold_z)  # False Negative Rate
        tpr = 1 - fnr  # True Positive Rate
        tnr = 1 - fpr  # True Negative Rate

        return {
            'fpr': fpr,
            'fnr': fnr,
            'tpr': tpr,
            'tnr': tnr,
            'mu0': mu0,
            'sigma0': sigma0,
            'mu_z0': np.mean(z_data0),
            'sigma_z0': np.std(z_data0),
            'mu_z1': np.mean(z_data1),
            'sigma_z1': np.std(z_data1),
            'z_data0': z_data0,
            'z_data1': z_data1
        }

    def plot_detection_distributions(self, stats: dict, threshold_z: float = 5.0,
                                   save_path: Optional[str] = None, show_plot: bool = True):

        # Extract statistics
        mu_z0, sigma_z0 = stats['mu_z0'], stats['sigma_z0']
        mu_z1, sigma_z1 = stats['mu_z1'], stats['sigma_z1']
        fpr, fnr = stats['fpr'], stats['fnr']

        z = np.linspace(-3, 10, 1000)
        pdf0 = norm.pdf(z, mu_z0, sigma_z0)
        pdf1 = norm.pdf(z, mu_z1, sigma_z1)

        # Calculate areas for FPR and FNR
        z_fpr = z[z > threshold_z]
        pdf_fpr = norm.pdf(z_fpr, mu_z0, sigma_z0)

        z_fnr = z[z <= threshold_z]
        pdf_fnr = norm.pdf(z_fnr, mu_z1, sigma_z1)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(z, pdf0, label='Original', color='#719F85')
        plt.fill_between(z_fpr, pdf_fpr, color='#719F85', alpha=0.3,
                        label=f'False Positive Rate ({fpr:.4f})')

        plt.plot(z, pdf1, label='Watermark', color='#F95551')
        plt.fill_between(z_fnr, pdf_fnr, color='#F95551', alpha=0.3,
                        label=f'False Negative Rate ({fnr:.4f})')

        plt.axvline(threshold_z, color='#4683F5', linestyle='--')

        plt.xlabel('z-score')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        max_y = max(max(pdf0), max(pdf1))
        plt.ylim(bottom=0, top=max_y * 1.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_results_csv(self, prompts: List[str], generated_texts: List[str],
                        match_ratios: List[float], trimmed_lengths: List[int], output_path: str):
        """
        Save generation results to CSV file.

        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            match_ratios: List of watermark match ratios
            trimmed_lengths: List of trimmed sequence lengths
            output_path: Path to save CSV file
        """
        fieldnames = ["prompt", "generated", "match_ratio", "trimmed_length"]

        # Create CSV if it doesn't exist
        if not os.path.exists(output_path):
            with open(output_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        # Append results
        with open(output_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for prompt, generated, ratio, length in zip(prompts, generated_texts, match_ratios, trimmed_lengths):
                writer.writerow({
                    "prompt": prompt,
                    "generated": generated,
                    "match_ratio": ratio,
                    "trimmed_length": length
                })

    def load_results_csv(self, csv_path: str, min_length_for_analysis: int = 200) -> Tuple[List[float], List[int]]:

        match_ratios = []
        trimmed_lengths = []

        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                length = int(row['trimmed_length'])
                ratio = float(row['match_ratio'])

                if length >= min_length_for_analysis:
                    match_ratios.append(ratio)
                    trimmed_lengths.append(length)

        return match_ratios, trimmed_lengths

    def analyze_batch_results(self, generated_sequences: List[List[int]], prompt_lengths: List[int],
                            min_length_for_analysis: int = 200) -> Tuple[List[float], List[int], List[int]]:

        match_ratios = []
        trimmed_lengths = []
        valid_ratios = []

        for seq, prompt_len in zip(generated_sequences, prompt_lengths):
            matched_count, total_length, match_ratio = self.calculate_match_ratio(seq, prompt_len)

            # Calculate trimmed length (up to EOT)
            try:
                cutoff_index = seq.index(self.eot_token_id)
                trimmed_length = cutoff_index + 1
            except ValueError:
                trimmed_length = len(seq)

            match_ratios.append(match_ratio)
            trimmed_lengths.append(trimmed_length)

            # Only include ratios for sequences above minimum length
            if trimmed_length >= min_length_for_analysis:
                valid_ratios.append(match_ratio)

        return match_ratios, trimmed_lengths, valid_ratios

    def print_detection_summary(self, stats: dict, threshold_z: float = 5.0):
        """Print a summary of detection statistics."""
        print(f"\n=== Watermark Detection Summary ===")
        print(f"Detection Threshold (z-score): {threshold_z}")
        print(f"False Positive Rate: {stats['fpr']:.4f}")
        print(f"False Negative Rate: {stats['fnr']:.4f}")
        print(f"True Positive Rate: {stats['tpr']:.4f}")
        print(f"True Negative Rate: {stats['tnr']:.4f}")
        print(f"\nWatermarked Text Statistics:")
        print(f"  Z-score Mean: {stats['mu_z1']:.2f}, Std: {stats['sigma_z1']:.2f}")
        print("=" * 40)