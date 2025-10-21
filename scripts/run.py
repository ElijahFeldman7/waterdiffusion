"""
LLaDA Model Runner Script

Usage examples:

1. Run with config file:
   python scripts/run.py --config configs/llada_8b_instruct.yaml --prompt "Hello, world!"

2. Run with explicit parameters:
   python scripts/run.py --model_path GSAI-ML/LLaDA-8B-Base --prompt "What is AI?" --steps 128 --gen_length 256

3. Run with custom generation settings:
   python scripts/run.py --config configs/llada_1_5.yaml --temperature 0.7 --cfg_scale 1.0

4. Quick test with defaults:
   python scripts/run.py --prompt "Explain quantum computing"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llada import LLaDAGenerator


def parse_args():
    """
    Parse command line arguments for model generation.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run LLaDA model generation")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["llada"],
        default="llada",
        help="Model type to use (currently only 'llada' supported)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="HuggingFace model path (overrides config)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (overrides config)"
    )
    
    parser.add_argument(
        "--gen_length",
        type=int,
        default=None,
        help="Length of text to generate in tokens (overrides config)"
    )
    
    parser.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Block length for generation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)"
    )
    
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale"
    )
    
    parser.add_argument(
        "--remasking",
        type=str,
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Remasking strategy"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, overrides config)"
    )
    
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template formatting"
    )
    
    return parser.parse_args()


def load_model(args):
    """
    Load the appropriate model based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Model generator instance
    """
    if args.model == "llada":
        if args.config:
            print(f"Loading LLaDA model from config: {args.config}")
            generator = LLaDAGenerator(
                config=args.config,
                model_path=args.model_path,
                device=args.device
            )
        else:
            print(f"Loading LLaDA model with explicit parameters")
            model_path = args.model_path or "GSAI-ML/LLaDA-8B-Instruct"
            generator = LLaDAGenerator(
                model_path=model_path,
                device=args.device or "cuda"
            )
        return generator
    else:
        raise ValueError(f"Model type '{args.model}' not supported")


def main():
    """
    Main entry point for running model generation.
    """
    args = parse_args()
    
    print("="*60)
    print("LLaDA Model Runner")
    print("="*60)
    
    generator = load_model(args)
    
    print(f"\nModel: {generator.model_path}")
    print(f"Device: {generator.device}")
    print(f"Default generation length: {generator.default_gen_length}")
    print(f"Default diffusion steps: {generator.default_steps}")
    print(f"Default temperature: {generator.default_temperature}")
    
    print("\n" + "-"*60)
    print(f"Prompt: {args.prompt}")
    print("-"*60)
    
    print("\nGenerating response...")
    
    response = generator.generate_text(
        prompt=args.prompt,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        use_chat_template=not args.no_chat_template
    )
    
    print("\n" + "="*60)
    print("Generated Response:")
    print("="*60)
    print(response)
    print("="*60)


if __name__ == "__main__":
    main()
