import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TPR @ 1%% FPR from subtext p-value details JSONL by attack ratio"
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=Path("runs/subtext_robustness_compare_04_20_04_44/subtext_pvalues_details.jsonl"),
        help="Path to subtext_pvalues_details.jsonl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="P-value threshold corresponding to target FPR (default: 0.01)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path; default is next to input JSONL",
    )
    parser.add_argument(
        "--show_counts",
        action="store_true",
        help="Include positive counts in printed table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_jsonl.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else input_path.parent / "tpr_at_1fpr_by_attack_ratio.csv"
    )

    # key = (dataset, detector, attack_ratio)
    totals: Dict[Tuple[str, str, float], int] = defaultdict(int)
    positives: Dict[Tuple[str, str, float], int] = defaultdict(int)

    with input_path.open("r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc

            dataset = row.get("dataset")
            detector = row.get("detector")
            attack_ratio = row.get("attack_ratio")
            p_value = row.get("p_value")

            if dataset is None or detector is None or attack_ratio is None or p_value is None:
                continue

            try:
                ratio = float(attack_ratio)
                p = float(p_value)
            except (TypeError, ValueError):
                continue

            key = (str(dataset), str(detector), round(ratio, 6))
            totals[key] += 1
            if p <= args.threshold:
                positives[key] += 1

    rows = []
    for key in sorted(totals.keys(), key=lambda k: (k[0], k[1], k[2])):
        dataset, detector, ratio = key
        n = totals[key]
        n_pos = positives.get(key, 0)
        tpr = (n_pos / n) if n > 0 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "detector": detector,
                "attack_ratio": ratio,
                "n_subtexts": n,
                f"n_detected_p_le_{args.threshold:g}": n_pos,
                "tpr_at_1pct_fpr": tpr,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with output_csv.open("w", newline="") as f:
            f.write("dataset,detector,attack_ratio,n_subtexts,n_detected,tpr_at_1pct_fpr\n")

    print(f"Input: {input_path}")
    print(f"Threshold (1% FPR): p <= {args.threshold:g}")
    print(f"Saved CSV: {output_csv}")
    print()

    if not rows:
        print("No valid rows found in input JSONL.")
        return

    print("TPR @ 1% FPR by dataset/detector and attack ratio")
    print("=" * 72)

    current_group = None
    for row in rows:
        group = (row["dataset"], row["detector"])
        if group != current_group:
            if current_group is not None:
                print()
            print(f"dataset={group[0]} | detector={group[1]}")
            print("ratio\tTPR@1%FPR" + ("\t(n_detected/n_subtexts)" if args.show_counts else ""))
            current_group = group

        if args.show_counts:
            count_col = f"n_detected_p_le_{args.threshold:g}"
            print(
                f"{row['attack_ratio']:.3f}\t{row['tpr_at_1pct_fpr']:.4f}\t"
                f"({row[count_col]}/{row['n_subtexts']})"
            )
        else:
            print(f"{row['attack_ratio']:.3f}\t{row['tpr_at_1pct_fpr']:.4f}")


if __name__ == "__main__":
    main()
