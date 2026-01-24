
import hashlib
import os
import json

class SimpleRemaskWatermark:

    @staticmethod
    def pair_rule_follows(key: str, token_a: int, token_b: int) -> bool:
        a = int(token_a)
        b = int(token_b)
        lo, hi = (a, b) if a < b else (b, a)
        h = hashlib.sha256()
        h.update(key.encode('utf-8'))
        h.update(b'|')
        if a == b:
            h.update(str(a).encode('utf-8'))
        else:
            h.update(str(lo).encode('utf-8'))
            h.update(b'|')
            h.update(str(hi).encode('utf-8'))
        v = int.from_bytes(h.digest()[:2], 'big')
        bit = v & 1
        if a == b:
            return bit
        if a < b:
            return 1 if bit == 1 else 0
        return 1 if bit == 0 else 0

    def __init__(
        self,
        key: str = "my_secret_key",
        delta: float = 2.0,
        context_window: list[int] = None,
    ):
        self.key = key
        self.delta = delta
        self.context_window = context_window or [-2, -1, 1, 2]

def generate_visualization_data(text, watermark, mask_ratio=0.5):
    """Generates a JSON file with data for the watermark visualization.
    
    Args:
        text: Input text to visualize
        watermark: Watermark instance
        mask_ratio: Fraction of tokens to mark as masked (default 0.5)
    """
    tokens = text.split()
    
    # Simulate masking - alternate masked/unmasked or random
    # For simplicity, let's make every other token masked
    import random
    random.seed(42)  # Reproducible
    is_masked = [random.random() < mask_ratio for _ in tokens]
    
    # Initialize confidences (all start at 0)
    confidences = [0.0] * len(tokens)
    
    data = {
        "text": text,
        "tokens": [],
        "steps": [],
        "delta": watermark.delta
    }

    for i, token_text in enumerate(tokens):
        data["tokens"].append({
            "text": token_text,
            "id": i,
            "is_masked": is_masked[i]
        })

    for i, proposed_token_str in enumerate(tokens):
        if not is_masked[i]:
            # Skip unmasked tokens in generation
            continue
            
        proposed_token_id = i
        
        step_data = {
            "step": i + 1,
            "token_id": proposed_token_id,
            "token_text": proposed_token_str,
            "evaluations": [],
            "total_satisfied": 0,
            "confidence_boost": 0.0
        }

        count = 0.0
        
        for j in watermark.context_window:
            neighbor_index = i + j
            if 0 <= neighbor_index < len(tokens):
                other_token_str = tokens[neighbor_index]
                other_token_id = neighbor_index
                
                # Determine contribution based on masked/unmasked
                contribution = confidences[neighbor_index] if is_masked[neighbor_index] else 1.0
                
                rule_followed = False
                if j < 0:
                    rule_followed = SimpleRemaskWatermark.pair_rule_follows(watermark.key, other_token_id, proposed_token_id)
                else:
                    rule_followed = SimpleRemaskWatermark.pair_rule_follows(watermark.key, proposed_token_id, other_token_id)
                
                step_data["evaluations"].append({
                    "pair": (tokens[min(i,neighbor_index)], tokens[max(i,neighbor_index)]),
                    "satisfied": rule_followed,
                    "context_token_id": other_token_id,
                    "is_masked": is_masked[neighbor_index],
                    "contribution": contribution if rule_followed else 0.0
                })

                if rule_followed:
                    count += contribution
        
        step_data["total_count"] = count
        step_data["confidence_boost"] = count * watermark.delta
        step_data["total_satisfied"] = sum(1 for e in step_data["evaluations"] if e["satisfied"])
        data["steps"].append(step_data)
        
        # Update confidence for this token
        confidences[i] += step_data["confidence_boost"]

    with open("visualization_data.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Generated visualization_data.json")


if __name__ == "__main__":
    watermark = SimpleRemaskWatermark()
    sample_text = "if we wanted to make our watermark a simulation for visualizing could we code this up for generation"
    generate_visualization_data(sample_text, watermark)

