import json
import sys
import numpy as np
from scipy.stats import zscore

# Usage: python watermark_strength_stats.py path/to/results.jsonl [field]
# Default field: det_wm_score
# For z-score stats, use e.g. det_wm_score or det_wm_satisfaction_rate

Z_THRESHOLDS = [(2.33, '99%'), (3.09, '99.9%')]


def main():
    if len(sys.argv) < 2:
        print("Usage: python watermark_strength_stats.py path/to/results.jsonl [field]")
        sys.exit(1)
    path = sys.argv[1]
    field = sys.argv[2] if len(sys.argv) > 2 else "det_wm_score"

    values = []
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                if field in row:
                    values.append(row[field])
            except Exception:
                continue
    values = np.array(values)
    zs = zscore(values)
    mean_z = np.mean(zs)
    std_z = np.std(zs)

    # Detection confidence
    stats = []
    for thresh, label in Z_THRESHOLDS:
        pct = 100.0 * np.mean(zs >= thresh)
        stats.append((label, thresh, pct))

    print("--- Watermark Strength (on watermarked samples) ---")
    print(f"Mean z-score: {mean_z:.4f} | Std. dev: {std_z:.4f}")
    conf_str = " | ".join([f"Detection Confidence: ≥{label} (z≥{thresh:.2f}): {pct:.2f}%" for label, thresh, pct in stats])
    print(conf_str)

    # Optionally, write to a file
    with open("watermark_strength_stats.txt", "w") as out:
        out.write("--- Watermark Strength (on watermarked samples) ---\n")
        out.write(f"Mean z-score: {mean_z:.4f} | Std. dev: {std_z:.4f}\n")
        out.write("Detection Confidence: " +
                  " | ".join([f"≥{label} (z≥{thresh:.2f}): {pct:.2f}%" for label, thresh, pct in stats]) + "\n")

if __name__ == "__main__":
    main()
