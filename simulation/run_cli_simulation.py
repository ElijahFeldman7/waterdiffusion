import torch
import numpy as np
import sys
import os
import random
import time

# Add the parent directory to the path to import the watermark module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from watermarks.watermark_original import SimpleRemaskWatermark

def softmax(x, dim):
    """Numerically stable softmax."""
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)

def create_fake_logits(tokens, full_vocab):
    """Creates a realistic-looking fake logit tensor for the full vocabulary."""
    num_tokens = len(tokens)
    vocab_map = {token: i for i, token in enumerate(full_vocab)}
    vocab_size = len(full_vocab)
    logits = torch.FloatTensor(1, num_tokens, vocab_size).uniform_(-3.5, -2.5)

    for i, token in enumerate(tokens):
        if token in vocab_map:
            logits[0, i, vocab_map[token]] = random.uniform(-1.5, -0.5)
    
    return logits, vocab_map

def print_step_details(step_num, total_steps, pos_to_unmask, evaluation):
    """Formats and prints the details of a single diffusion step to the console."""
    original_text = evaluation['original_text']
    chosen_text = evaluation['chosen_text']
    candidates = sorted(evaluation['candidates'], key=lambda x: x['final_prob'], reverse=True)
    
    biased_candidate = next((c for c in evaluation['candidates'] if c['final_logit'] > c['base_logit']), None)

    print("=" * 70)
    print(f" Diffusion Step {step_num} / {total_steps}: Generating for position {pos_to_unmask}")
    print("=" * 70)
    
    if biased_candidate:
        bias = biased_candidate['final_logit'] - biased_candidate['base_logit']
        print(f" -> Watermark bias of {bias:+.2f} applied to candidate: '{biased_candidate['text']}'")
    else:
        print(" -> No positive watermark bias was applied (model's top choice was already 'green').")

    if original_text != chosen_text:
        print(f" -> RESULT: Watermark CHANGED the output from '{original_text}' to '{chosen_text}'!")
    else:
        print(f" -> RESULT: Watermark did NOT change the output. Model's choice was '{original_text}'.")
        
    print("-" * 70)
    print(" Logit Inspector:")
    print("  {:<12} | {:<10} | {:<10} | {}".format("Candidate", "Base Prob", "Final Prob", "Change"))
    print("  " + "-" * 65)

    for cand in candidates[:5]:
        change = cand['final_prob'] - cand['base_prob']
        change_str = f"({change:+.3f})"
        marker = ""
        if cand['text'] == original_text: marker += " [Original]"
        if cand['text'] == chosen_text: marker += " [CHOSEN]"
        print("  {:<12} |   {:<8.3f} |   {:<8.3f} | {:<10} {}".format(cand['text'], cand['base_prob'], cand['final_prob'], change_str, marker))
    
    print()
    time.sleep(1)

def run_generative_simulation(text, watermark, num_masked=15):
    """
    Runs a true generative simulation with dynamic logit updates.
    """
    original_tokens = text.split()
    num_tokens = len(original_tokens)
    full_vocab = list(set(original_tokens))
    
    random.seed(42)
    masked_indices = list(range(3, num_tokens))
    random.shuffle(masked_indices)
    masked_indices = masked_indices[:num_masked]
    
    # This represents the true, generated sequence
    generated_token_indices = [None] * num_tokens
    
    logits, vocab_map = create_fake_logits(original_tokens, full_vocab)
    vocab_list = [t for t, i in sorted(vocab_map.items(), key=lambda item: item[1])]

    # Initialize with the prompt
    for i in range(3):
        generated_token_indices[i] = vocab_map[original_tokens[i]]

    print("Starting generative simulation with Dynamic Logit Updates...")
    print(f"Initial context: \"{' '.join(original_tokens[:3])}\"")
    
    for i, pos_to_unmask in enumerate(masked_indices):
        # The context for the watermark is the text generated so far.
        x0_context = torch.tensor([g if g is not None else -1 for g in generated_token_indices]).unsqueeze(0)
        
        biased_logits = watermark.bias_logits(logits=logits.clone(), x0_context=x0_context)
        
        base_logits_slice = logits[0, pos_to_unmask]
        biased_logits_slice = biased_logits[0, pos_to_unmask]
        biased_probs_slice = softmax(biased_logits_slice, dim=-1)
        
        chosen_token_idx = torch.argmax(biased_probs_slice).item()
        chosen_token_text = vocab_list[chosen_token_idx]
        
        original_top_idx = torch.argmax(base_logits_slice).item()
        original_top_text = vocab_list[original_top_idx]

        evaluation_data = {
            "original_text": original_top_text,
            "chosen_text": chosen_token_text,
            "candidates": [{
                "text": token,
                "base_logit": base_logits_slice[j].item(),
                "final_logit": biased_logits_slice[j].item(),
                "base_prob": softmax(base_logits_slice, dim=-1)[j].item(),
                "final_prob": biased_probs_slice[j].item(),
            } for j, token in enumerate(vocab_list)]
        }
        
        print_step_details(i + 1, len(masked_indices), pos_to_unmask, evaluation_data)

        # --- DYNAMIC LOGIT UPDATE ---
        generated_token_indices[pos_to_unmask] = chosen_token_idx
        
        # Now, update the logits of *other* tokens based on this new choice
        # This simulates the model re-evaluating its beliefs.
        update_neighborhood = watermark._get_context_window(pos_to_unmask)
        for neighbor_offset in update_neighborhood:
            neighbor_pos = pos_to_unmask + neighbor_offset
            if 0 <= neighbor_pos < num_tokens and generated_token_indices[neighbor_pos] is None:
                # This is a future, un-generated token. Let's update its beliefs.
                # For every word in the vocab, adjust its logit based on its pairing with the new token.
                for vocab_idx, token_text in enumerate(vocab_list):
                    color = watermark.get_pair_color(watermark.key, chosen_token_idx, vocab_idx)
                    # Apply a small, secondary bias to simulate context update
                    secondary_bias = color * watermark.delta * 0.1 
                    logits[0, neighbor_pos, vocab_idx] += secondary_bias

    final_generated_tokens = [vocab_list[idx] if idx is not None else "[MASK]" for idx in generated_token_indices]
    
    print("=" * 70)
    print("Simulation Complete.")
    print(f"Original Text:  \"{text}\"")
    print(f"Generated Text: \"{' '.join(final_generated_tokens)}\"")


if __name__ == "__main__":
    watermark = SimpleRemaskWatermark(key="my_secret_key", delta=4.0, num_neighbors=4)
    sample_text = "if we wanted to make our watermark a simulation for visualizing could we code this up for generation"
    run_generative_simulation(sample_text, watermark)
