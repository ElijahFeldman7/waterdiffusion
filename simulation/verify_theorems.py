import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import math

# Add the parent directory to the path to import the watermark module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from watermarks.watermark_original import SimpleRemaskWatermark

# --- Simulation Parameters ---
VOCAB_SIZE = 10000
SEQ_LEN = 256
BATCH_SIZE = 1
DELTA = 2.0
NUM_NEIGHBORS = 4
P_VALUE_THRESHOLD = 0.05

# --- Test Functions ---

def test_kl_divergence(watermark):
    """
    Tests Theorem 1: Selection-Step KL Divergence (for the updated, linear-bias algorithm).
    """
    print("\n--- Verifying Theorem 1: KL Divergence (Linear Bias) ---")
    
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    x0_context = torch.argmax(logits, dim=-1)
    
    biased_logits = logits.clone()
    kl_divergences = []
    
    examples_shown = 0
    
    for i in range(SEQ_LEN):
        proposed_token = int(x0_context[0, i].item())
        count = 0.0
        context_window = watermark._get_context_window(i)

        for j in context_window:
            neighbor_index = i + j
            if 0 <= neighbor_index < SEQ_LEN:
                other_token = int(x0_context[0, neighbor_index].item())
                # In the current algorithm, contribution is always 1.0
                weight = 1.0
                
                # Check rule based on neighbor position
                if (j < 0 and SimpleRemaskWatermark.pair_rule_follows(watermark.key, other_token, proposed_token)) or \
                   (j > 0 and SimpleRemaskWatermark.pair_rule_follows(watermark.key, proposed_token, other_token)):
                    count += weight
        
        original_logit_slice = logits[0, i]
        
        if count > 0:
            # Using the linear bias from the updated algorithm
            bias = count * watermark.delta
            biased_logits[0, i, proposed_token] += bias
        
        biased_logit_slice = biased_logits[0, i]
        
        kl_div = F.kl_div(F.log_softmax(biased_logit_slice, dim=-1), F.softmax(original_logit_slice, dim=-1), reduction='sum')
        kl_divergences.append(kl_div.item())

        if examples_shown < 5 and count > 0:
            bias = count * watermark.delta
            print(f"  - Pos {i}: Found {count} satisfied pairs. Applying linear bias of {bias:.4f}. Resulting KL = {kl_div.item():.4f}")
            examples_shown += 1

    avg_kl = np.mean(kl_divergences) if kl_divergences else 0
    max_kl = np.max(kl_divergences) if kl_divergences else 0
    
    # Theorem 1 states D_KL <= delta * k
    theoretical_bound = watermark.delta * watermark.num_neighbors
    
    print("\n--- Summary ---")
    print(f"Average KL Divergence (Empirical): {avg_kl:.4f}")
    print(f"Maximum KL Divergence (Empirical): {max_kl:.4f}")
    print(f"Theoretical Bound (D_KL <= delta * k): {theoretical_bound:.4f}")
    
    # Add a small tolerance for floating point issues
    if max_kl <= theoretical_bound + 1e-6:
        print("Result: The empirical KL divergence is WITHIN the theoretical bound.")
    else:
        print("Result: The empirical KL divergence EXCEEDS the theoretical bound.")

def test_false_positive_rate(watermark, num_samples=500):
    """
    Tests Theorem 4: False Positive (Forgery) Probability.
    This function uses the updated 'detect' method.
    """
    print(f"\n--- Verifying Theorem 4: False Positive Rate (on {num_samples} samples) ---")
    
    class MockTokenizer:
        def encode(self, text, return_tensors):
            # Generate random tokens to simulate non-watermarked text
            return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    false_positives = 0
    for _ in range(num_samples):
        random_text = "This is a random text" # The content doesn't matter for the mock tokenizer
        detection_result = watermark.detect(random_text, tokenizer=MockTokenizer(), p_value_threshold=P_VALUE_THRESHOLD)
        if detection_result["is_watermarked"]:
            false_positives += 1
            
    empirical_fp_rate = false_positives / num_samples
    
    print("\n--- Summary ---")
    print(f"Empirical False Positive Rate: {empirical_fp_rate:.4f} (at p_value_threshold={P_VALUE_THRESHOLD})")
    print("Analysis: This rate should be close to the p_value_threshold if the null hypothesis (p=0.5) is correct.")
    # Check if the empirical rate is reasonably close to the target threshold
    if abs(empirical_fp_rate - P_VALUE_THRESHOLD) < 0.04:
         print("Result: Empirical FP rate is reasonable. The statistical test appears correct.")
    else:
         print("Result: Empirical FP rate deviates significantly from the p-value threshold.")

if __name__ == "__main__":
    print("Running watermark theorem verification for the CURRENT algorithm...")
    
    watermark = SimpleRemaskWatermark(
        key="test_key",
        delta=DELTA,
        num_neighbors=NUM_NEIGHBORS,
        context_range=30  # context_range for the test
    )
    
    test_kl_divergence(watermark)
    test_false_positive_rate(watermark)

    print("\nVerification finished.")
