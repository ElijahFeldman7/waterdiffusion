import json
import math
import sys
from pathlib import Path

import numpy as np

from scripts.text_quality import evaluate_text_quality


def analyze_jsonl(file_path: Path, output_txt_path: Path):
    report_lines = []
    sample_details = []
    total_samples = 0
    
    tp, fp, tn, fn = 0, 0, 0, 0

    watermark_strength_stats = {
        "p_values": [], "z_scores": [], "satisfaction_rates": [],
        "num_timestamps": [], "unmasked_tokens_per_timestamp": [], "detection_margins": [],
    }
    
    quality_stats = {
        "gen_time_wm": [], "gen_time_no_wm": [], "perplexity_wm": [],
        "perplexity_no_wm": [], "cosine_similarity": [],
    }

    quality_totals_wm = {k: 0.0 for k in ["avg_sentence_length", "avg_word_length", "lexical_diversity", "flesch_reading_ease", "sequence_length"]}
    quality_totals_no_wm = {k: 0.0 for k in quality_totals_wm}
    quality_sample_count = 0

    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                
                try:
                    result = json.loads(line)
                    total_samples += 1
                    
                    if 'detection_wm' in result:
                        if result['detection_wm'].get('is_watermarked', False):
                            tp += 1
                        else:
                            fn += 1

                    if 'detection_no_wm' in result:
                        if result['detection_no_wm'].get('is_watermarked', False):
                            fp += 1
                        else:
                            tn += 1
                    
                    detection_wm = result.get('detection_wm', {})
                    if detection_wm:
                        if (p_value := detection_wm.get('score')) is not None: watermark_strength_stats['p_values'].append(p_value)
                        if (z_score := detection_wm.get('z_score')) is not None:
                            watermark_strength_stats['z_scores'].append(z_score)
                        if (sr := detection_wm.get('satisfaction_rate')) is not None: watermark_strength_stats['satisfaction_rates'].append(sr)
                        if (nt := detection_wm.get('num_timestamps')) is not None: watermark_strength_stats['num_timestamps'].append(nt)
                        if (ut := detection_wm.get('unmasked_tokens_per_timestamp')) is not None: watermark_strength_stats['unmasked_tokens_per_timestamp'].append(ut)
                        if p_value is not None and (thresh := detection_wm.get('threshold')) is not None:
                            watermark_strength_stats['detection_margins'].append(thresh - p_value)

                    for key in quality_stats:
                        if (val := result.get(key)) is not None and not math.isnan(val):
                            quality_stats[key].append(val)

                    if (text_wm := result.get('generated_text_wm')) and (text_no_wm := result.get('generated_text_no_wm')):
                        quality_metrics_wm = evaluate_text_quality(text_wm)
                        quality_metrics_no_wm = evaluate_text_quality(text_no_wm)
                        quality_sample_count += 1
                        for key in quality_totals_wm:
                            if (val := quality_metrics_wm.get(key)) is not None and not math.isnan(val): quality_totals_wm[key] += val
                            if (val := quality_metrics_no_wm.get(key)) is not None and not math.isnan(val): quality_totals_no_wm[key] += val
                    
                    sample_details.append(f"--- Sample {i+1} ---")
                    sample_details.append(f"Prompt: {result.get('prompt', 'N/A')}")
                    sample_details.append(f"Perplexity (WM): {result.get('perplexity_wm', 'N/A'):.4f} | Perplexity (No WM): {result.get('perplexity_no_wm', 'N/A'):.4f}")
                    sample_details.append(f"Z-score (WM): {detection_wm.get('z_score', 'N/A'):.4f}")
                    
                    wm_status = "True Positive" if detection_wm.get('is_watermarked', False) else "False Negative"
                    no_wm_status = "False Positive" if detection_no_wm.get('is_watermarked', False) else "True Negative"
                    sample_details.append(f"Watermarked Text Detection: {wm_status}")
                    sample_details.append(f"Non-Watermarked Text Detection: {no_wm_status}")

                    sample_details.append("\n ")
                    sample_details.append("\n[Watermarked Text]")
                    sample_details.append(result.get('generated_text_wm', 'N/A'))
                    sample_details.append("\n[Original Text]")
                    sample_details.append(result.get('generated_text_no_wm', 'N/A'))
                    sample_details.append("\n")


                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

        if total_samples == 0:
            print("Error: No valid results found in the file.")
            return

        final_results = {"summary": {}, "detection_rates": {}, "watermark_strength": {}, "performance_impact": {}, "quality_impact": {}, "detailed_quality_metrics": {}}

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        final_results["summary"] = {
            "total_samples": total_samples, 
            "true_positives": tp, "false_negatives": fn,
            "true_negatives": tn, "false_positives": fp
        }
        final_results["detection_rates"] = {
            "tpr": tpr, "fpr": fpr, "fnr": fnr, "tnr": tnr
        }

        report_lines.append("--- Watermark Detection Analysis ---")
        report_lines.append(f"Total samples processed: {total_samples}")
        report_lines.append(f"True Positives (TP): {tp}")
        report_lines.append(f"False Negatives (FN): {fn}")
        report_lines.append(f"True Negatives (TN): {tn}")
        report_lines.append(f"False Positives (FP): {fp}")
        report_lines.append("\n")
        report_lines.append("--- Detection Rates ---")
        report_lines.append(f"True Positive Rate (TPR/Recall): {tpr:.4f}")
        report_lines.append(f"False Positive Rate (FPR): {fpr:.4f}")
        report_lines.append(f"False Negative Rate (FNR): {fnr:.4f}")
        report_lines.append(f"True Negative Rate (TNR/Specificity): {tnr:.4f}")
        report_lines.append("\n")

        if watermark_strength_stats['z_scores']:
            mean_z = float(np.mean(watermark_strength_stats['z_scores']))
            std_z = float(np.std(watermark_strength_stats['z_scores']))
            strong_detection_count = sum(1 for z in watermark_strength_stats['z_scores'] if z >= 2.326)
            very_strong_detection_count = sum(1 for z in watermark_strength_stats['z_scores'] if z >= 3.090)
            strong_rate = (strong_detection_count / len(watermark_strength_stats['z_scores'])) * 100 if watermark_strength_stats['z_scores'] else 0
            very_strong_rate = (very_strong_detection_count / len(watermark_strength_stats['z_scores'])) * 100 if watermark_strength_stats['z_scores'] else 0
            final_results["watermark_strength"] = {
                "mean_z_score": mean_z, "std_dev_z_score": std_z,
                "strong_detection_rate_percent": strong_rate, "very_strong_detection_rate_percent": very_strong_rate
            }
            report_lines.append("--- Watermark Strength (on watermarked samples) ---")
            report_lines.append(f"Mean z-score: {mean_z:.4f} | Std. dev: {std_z:.4f}")
            report_lines.append(f"Detection Confidence: ≥99% (z≥2.33): {strong_rate:.2f}% | ≥99.9% (z≥3.09): {very_strong_rate:.2f}%")
            report_lines.append("\n")

        if quality_stats["gen_time_wm"]:
            mean_gen_time_wm = float(np.mean(quality_stats["gen_time_wm"]))
            mean_gen_time_no_wm = float(np.mean(quality_stats["gen_time_no_wm"]))
            time_diff = mean_gen_time_wm - mean_gen_time_no_wm
            time_overhead = (time_diff / mean_gen_time_no_wm) * 100 if mean_gen_time_no_wm > 0 else 0
            final_results["performance_impact"] = {
                "mean_gen_time_wm_s": mean_gen_time_wm, "mean_gen_time_no_wm_s": mean_gen_time_no_wm,
                "mean_time_overhead_s": time_diff, "mean_time_overhead_percent": time_overhead
            }
            report_lines.append("--- Performance Impact ---")
            report_lines.append(f"Generation Time (WM): {mean_gen_time_wm:.4f}s")
            report_lines.append(f"Generation Time (No WM): {mean_gen_time_no_wm:.4f}s")
            report_lines.append(f"Mean Time Overhead: {time_diff:.4f}s ({time_overhead:.2f}%)")
            report_lines.append("\n")

        if quality_stats["perplexity_wm"]:
            mean_ppl_wm = float(np.mean(quality_stats["perplexity_wm"]))
            mean_ppl_no_wm = float(np.mean(quality_stats["perplexity_no_wm"]))
            ppl_diff = mean_ppl_wm - mean_ppl_no_wm
            ppl_increase = (ppl_diff / mean_ppl_no_wm) * 100 if mean_ppl_no_wm > 0 else 0
            mean_cos_sim = float(np.mean(quality_stats["cosine_similarity"]))
            final_results["quality_impact"] = {
                "mean_perplexity_wm": mean_ppl_wm, "mean_perplexity_no_wm": mean_ppl_no_wm,
                "mean_perplexity_increase": ppl_diff, "mean_perplexity_increase_percent": ppl_increase,
                "mean_cosine_similarity": mean_cos_sim
            }
            report_lines.append("--- Quality Impact ---")
            report_lines.append(f"Perplexity (WM): {mean_ppl_wm:.4f}")
            report_lines.append(f"Perplexity (No WM): {mean_ppl_no_wm:.4f}")
            report_lines.append(f"Mean Perplexity Increase: {ppl_diff:.4f} ({ppl_increase:.2f}%)")
            report_lines.append(f"Cosine Similarity (WM vs No WM): {mean_cos_sim:.4f}")
            report_lines.append("\n")

        if quality_sample_count > 0:
            report_lines.append("--- Detailed Text Quality Metrics (averages) ---")
            header = f"{ 'Metric':<25} | {'Watermarked':<15} | {'No Watermark':<15}"
            report_lines.append(header)
            report_lines.append("-" * len(header))
            detailed_wm = {}
            detailed_no_wm = {}
            for key in quality_totals_wm:
                avg_wm = quality_totals_wm[key] / quality_sample_count
                avg_no_wm = quality_totals_no_wm[key] / quality_sample_count
                detailed_wm[key] = avg_wm
                detailed_no_wm[key] = avg_no_wm
                report_lines.append(f"{key.replace('_', ' ').title():<25} | {avg_wm:<15.4f} | {avg_no_wm:<15.4f}")
            final_results["detailed_quality_metrics"] = {"watermarked": detailed_wm, "no_watermark": detailed_no_wm}
            report_lines.append(f"\nSamples evaluated for quality: {quality_sample_count}")
            report_lines.append("\n")

        results_json_path = file_path.parent / "results.json"
        with open(results_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        summary_report_content = "\n".join(report_lines)
        full_report_content = summary_report_content + "\n--- Generation Samples ---\\n" + "\n".join(sample_details)
        
        with open(output_txt_path, 'w') as f:
            f.write(full_report_content)

        print(summary_report_content)
        print(f"\nAggregated results saved to: {results_json_path}")
        print(f"Full detailed report saved to: {output_txt_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        sys.exit(1)
