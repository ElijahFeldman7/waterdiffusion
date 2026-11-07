#!/usr/bin/env python3
import argparse
import torch
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import setup_c4_dataset
from generation import WatermarkGenerator
from detection import WatermarkDetector
from utils import calculate_watermark_stats


def setup_model_and_tokenizer(model_name: str = 'GSAI-ML/LLaDA-1.5', device: str = 'cuda', cache_dir: str = '/dev/shm'):
    """Setup model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device).eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with watermarking")

    # Generation method
    parser.add_argument('--method', choices=['original', 'watermark', 'beam'],
                       default='beam', help='Generation method: original (no watermark), watermark (direct), beam (with evaluation)')

    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples to generate')
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--block_length', type=int, default=32, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--remasking', choices=['low_confidence', 'random'], default='low_confidence')

    # Method-specific parameters
    parser.add_argument('--top_k', type=int, default=3, help='Top-k for multinomial sampling')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search watermarking')
    parser.add_argument('--sampling_strategy', choices=['greedy', 'multinomial'], default='greedy',
                       help='Token sampling strategy for watermark and beam methods')
    parser.add_argument('--enable_watermark', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--private_key', help='Private key for watermarking (can be numeric or string, adds randomness to parity mapping)')

    # Model and data
    parser.add_argument('--model_name', default='GSAI-ML/LLaDA-1.5', help='Model name or path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache_dir', default='/dev/shm', help='Model cache directory')
    parser.add_argument('--dataset_path', default='c4-validation.00000-of-00001.json.gz',
                       help='Path to dataset file')
    parser.add_argument('--dataset_url', help='URL to download dataset from')

    # Output
    parser.add_argument('--output_prefix', default='generated_results', help='Output file prefix')
    parser.add_argument('--min_length_for_analysis', type=int, default=200,
                       help='Minimum sequence length for ratio analysis')

    args = parser.parse_args()

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device, args.cache_dir)
    generator = WatermarkGenerator(model, tokenizer, args.device, private_key=args.private_key)

    # Setup data
    if args.dataset_url:
        prompts = setup_c4_dataset(args.dataset_url, args.dataset_path)
    else:
        prompts = setup_c4_dataset(dataset_path=args.dataset_path)

    if args.num_samples:
        prompts = prompts[:args.num_samples]

    detector = WatermarkDetector()

    # Setup output files
    output_prefix = f"{args.output_prefix}_{args.method}"
    if args.method == 'multinomial':
        output_prefix += f"_k{args.top_k}"
    elif args.method == 'beam':
        output_prefix += f"_{args.sampling_strategy}_beam{args.beam_size}"
        if args.sampling_strategy == 'multinomial':
            output_prefix += f"_k{args.top_k}"
        if not args.enable_watermark:
            output_prefix += "_nowm"

    csv_file = f"{output_prefix}.csv"

    # Clear existing CSV file
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Print generation configuration
    print(f"\n=== Generation Configuration ===")
    print(f"Method: {args.method}")
    print(f"Samples: {args.num_samples}")
    print(f"Generation length: {args.gen_length}")
    print(f"Block length: {args.block_length}")
    if args.method == 'multinomial':
        print(f"Top-k: {args.top_k}")
    elif args.method == 'beam':
        print(f"Beam size: {args.beam_size}")
        if args.beam_size == 1:
            print(f"Note: Beam size 1 - using direct {args.sampling_strategy} method for efficiency")
        print(f"Sampling strategy: {args.sampling_strategy}")
        if args.sampling_strategy == 'multinomial':
            print(f"Top-k: {args.top_k}")
        print(f"Watermarking: {'Enabled' if args.enable_watermark else 'Disabled'}")
    if args.private_key:
        print(f"Private key: ***provided***")
    else:
        print(f"Private key: None (using position-based watermarking)")
    print(f"Output file: {csv_file}")
    print("=" * 35)

    all_ratios = []
    all_lengths = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str)['input_ids']
        input_ids_tensor = torch.tensor(input_ids).to(args.device).unsqueeze(0)

        generation_kwargs = {
            'prompt': input_ids_tensor,
            'steps': args.gen_length,
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'temperature': args.temperature,
            'cfg_scale': args.cfg_scale,
            'remasking': args.remasking
        }

        if args.method == 'original':
            output = generator.generate_original(**generation_kwargs)
        elif args.method == 'watermark':
            if args.sampling_strategy == 'greedy':
                output = generator.generate_watermark_greedy(**generation_kwargs)
            elif args.sampling_strategy == 'multinomial':
                output = generator.generate_watermark_multinomial(top_k=args.top_k, verbose=args.verbose, **generation_kwargs)
            else:
                raise ValueError(f"Unknown sampling strategy: {args.sampling_strategy}")
        elif args.method == 'beam':
            output = generator.generate_beam_search(beam_size=args.beam_size,
                                                   sampling_strategy=args.sampling_strategy,
                                                   top_k=args.top_k,
                                                   enable_watermark=args.enable_watermark,
                                                   **generation_kwargs)
        else:
            raise ValueError(f"Unknown generation method: {args.method}")

        generated_text = tokenizer.batch_decode(output[:, input_ids_tensor.shape[1]:], skip_special_tokens=True)[0]
        generated_ids = output[0, input_ids_tensor.shape[1]:].tolist()

        matched_count, total_length, match_ratio, trimmed_length = calculate_watermark_stats(
            generated_ids, input_ids_tensor.shape[1], private_key=args.private_key
        )

        all_ratios.append(match_ratio)
        all_lengths.append(trimmed_length)

        print(f"[{i+1}/{len(prompts)}] Matching ratio: {matched_count}/{trimmed_length} ({match_ratio * 100:.1f}%)")

        rounded_match_ratio = round(match_ratio, 4)
        detector.save_results_csv([prompt], [generated_text], [rounded_match_ratio], [trimmed_length], csv_file)

        if i % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\n=== Generation Summary ===")
    print(f"Results saved to: {csv_file}")
    print(f"Total sequences: {len(all_ratios)}")
    print(f"Sequences ≥{args.min_length_for_analysis} tokens: {len([l for l in all_lengths if l >= args.min_length_for_analysis])}")

    if all_ratios:
        print(f"Average match ratio: {sum(all_ratios)/len(all_ratios):.4f}")
        valid_ratios = [r for r, l in zip(all_ratios, all_lengths) if l >= args.min_length_for_analysis]
        if valid_ratios:
            print(f"Average match ratio (≥{args.min_length_for_analysis} tokens): {sum(valid_ratios)/len(valid_ratios):.4f}")
    print("=" * 27)


if __name__ == "__main__":
    main()