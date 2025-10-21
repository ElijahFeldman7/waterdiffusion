"""
Example script demonstrating watermark integration with LLaDA model.

This script shows how to:
1. Initialize a watermark instance
2. Create a LLaDA generator with watermarking enabled
3. Generate watermarked text

The watermark modifies the remasking strategy by controlling which tokens
get selected/unmasked at each diffusion step through confidence score manipulation.
"""

from models.llada import LLaDAGenerator
from watermarks.simple_remask_watermark import SimpleRemaskWatermark


def main():
    print("="*60)
    print("LLaDA Watermarking Example")
    print("="*60)
    
    watermark_config = {}
    secret_key = "my_secret_watermark_key_12345"
    delta = 2.0
    watermark = SimpleRemaskWatermark(config=watermark_config, key=secret_key, delta=delta)
    print(f"\nInitialized watermark: {watermark}")
    print(f"Using secret key: {secret_key}")
    print(f"Delta (confidence boost): {delta}")
    
    generator = LLaDAGenerator(
        config="configs/llada_8b_instruct.yaml",
        watermark=watermark
    )
    print(f"Initialized LLaDA generator with watermark")
    
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about technology."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'-'*60}")
        print(f"Example {i}")
        print(f"{'-'*60}")
        print(f"Prompt: {prompt}")
        print("\nGenerating watermarked response...")
        
        response = generator.generate_text(prompt)
        
        print(f"Response: {response}")
        
        detection_result = watermark.detect(
            response,
            generator.tokenizer
        )
        print(f"\nWatermark detection result: {detection_result}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()

