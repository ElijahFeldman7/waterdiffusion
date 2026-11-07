#!/usr/bin/env python3
import argparse
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection import WatermarkDetector


def main():
    parser = argparse.ArgumentParser(description="Detect watermarks using statistical analysis")

    # Required files
    parser.add_argument('--watermarked', required=True, help='CSV file with watermarked text results')
    parser.add_argument('--original', required=True, help='CSV file with original text results')

    # Detection parameters
    parser.add_argument('--threshold_z', type=float, default=4.0, help='Z-score threshold for detection (default: 4.0)')
    parser.add_argument('--min_length', type=int, default=200,
                       help='Minimum sequence length for analysis (default: 200)')
    parser.add_argument('--private_key', help='Private key for watermark detection (must match key used for generation)')

    # Output options
    parser.add_argument('--plot', help='Output path for detection plot (optional)')
    parser.add_argument('--plot_title', default='Watermark Detection Analysis')
    parser.add_argument('--show_plot', action='store_true')

    args = parser.parse_args()

    # Initialize detector
    detector = WatermarkDetector(private_key=args.private_key)

    print(f"=== Watermark Detection Analysis ===")
    print(f"Watermarked file: {args.watermarked}")
    print(f"Original file: {args.original}")
    print(f"Private key: {'***provided***' if args.private_key else 'None (using position-based detection)'}")
    print(f"Z-score threshold: {args.threshold_z}")
    print(f"Minimum length: {args.min_length}")
    print("=" * 37)

    # Load data from CSV files
    try:
        watermarked_ratios, watermarked_lengths = detector.load_results_csv(args.watermarked, args.min_length)
    except Exception as e:
        print(f"Error loading watermarked file: {e}")
        sys.exit(1)

    try:
        original_ratios, original_lengths = detector.load_results_csv(args.original, args.min_length)
    except Exception as e:
        print(f"Error loading original file: {e}")
        sys.exit(1)

    if not watermarked_ratios:
        print(f"Error: No valid watermarked sequences found (≥{args.min_length} tokens)")
        sys.exit(1)

    if not original_ratios:
        print(f"Error: No valid original sequences found (≥{args.min_length} tokens)")
        sys.exit(1)

    print(f"Loaded {len(watermarked_ratios)} watermarked sequences (≥{args.min_length} tokens)")
    print(f"Loaded {len(original_ratios)} original sequences (≥{args.min_length} tokens)")

    stats = detector.compute_detection_stats(watermarked_ratios, original_ratios, args.threshold_z)

    # Print detailed summary
    detector.print_detection_summary(stats, args.threshold_z)

    print(f"\nSequence Statistics:")
    print(f"Watermarked - Mean ratio: {sum(watermarked_ratios)/len(watermarked_ratios):.4f}, "
          f"Std: {(sum([(r - sum(watermarked_ratios)/len(watermarked_ratios))**2 for r in watermarked_ratios])/len(watermarked_ratios))**0.5:.4f}")
    print(f"Original - Mean ratio: {sum(original_ratios)/len(original_ratios):.4f}, "
          f"Std: {(sum([(r - sum(original_ratios)/len(original_ratios))**2 for r in original_ratios])/len(original_ratios))**0.5:.4f}")

    if args.plot or args.show_plot:
        print(f"\nGenerating detection plot...")
        detector.plot_detection_distributions(
            stats,
            args.threshold_z,
            args.plot,
            args.show_plot
        )
        if args.plot:
            print(f"Detection plot saved to: {args.plot}")

    print(f"\nDetection analysis complete!")


if __name__ == "__main__":
    main()