
import torch
import hashlib
import os

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

def visualize_watermark_process(text, watermark):
    """Generates an HTML visualization of the watermarking process."""
    # A pseudo-tokenizer for demonstration purposes
    tokens = text.split()
    token_ids = {token: i for i, token in enumerate(set(tokens))}
    token_indices = [token_ids[token] for token in tokens]

    html = """
    <html>
    <head>
        <title>Watermark Visualization</title>
        <style>
            body { font-family: sans-serif; }
            .token { border: 1px solid #ccc; padding: 5px; margin: 2px; display: inline-block; }
            .context { border: 2px solid blue; }
            .satisfied { background-color: #d4edda; } /* Green */
            .not-satisfied { background-color: #f8d7da; } /* Red */
            .step { margin-bottom: 20px; border-left: 3px solid #666; padding-left: 15px;}
        </style>
    </head>
    <body>
        <h1>Watermark Process Visualization</h1>
    """

    html += "<h2>Original Text:</h2><p>" + " ".join(f"<span class='token'>{token}({token_ids[token]})</span>" for token in tokens) + "</p>"

    for i, proposed_token_str in enumerate(tokens):
        proposed_token = token_ids[proposed_token_str]
        html += f"<div class='step'><h3>Step {i+1}: Evaluating token '{proposed_token_str}' ({proposed_token})</h3>"
        
        count = 0.0
        
        # Display the context window
        html += "<div>"
        for j, token_str in enumerate(tokens):
            extra_class = "context" if abs(i-j) in map(abs,watermark.context_window) and i!=j else ""
            html += f"<span class='token {extra_class}'>{token_str}({token_ids[token_str]})</span>"
        html += "</div><br/>"


        for j in watermark.context_window:
            neighbor_index = i + j
            if 0 <= neighbor_index < len(tokens):
                other_token_str = tokens[neighbor_index]
                other_token = token_ids[other_token_str]
                
                rule_followed = False
                if j < 0:
                    rule_followed = SimpleRemaskWatermark.pair_rule_follows(watermark.key, other_token, proposed_token)
                else:
                    rule_followed = SimpleRemaskWatermark.pair_rule_follows(watermark.key, proposed_token, other_token)
                
                status_class = "satisfied" if rule_followed else "not-satisfied"
                html += f"<div class='{status_class}'>Checking pair: ('{tokens[min(i,neighbor_index)]}', '{tokens[max(i,neighbor_index)]}'). Rule satisfied: {rule_followed}.</div>"
                
                if rule_followed:
                    count += 1
        
        confidence_boost = count * watermark.delta
        html += f"<p>Total satisfied pairs: {int(count)}</p>"
        html += f"<p>Confidence boost for '{proposed_token_str}': {count} * {watermark.delta} = {confidence_boost}</p>"
        html += "</div>"

    html += """
    </body>
    </html>
    """
    
    with open("watermark_visualization.html", "w") as f:
        f.write(html)
    print("Generated watermark_visualization.html")


if __name__ == "__main__":
    watermark = SimpleRemaskWatermark()
    # sample_text = "The quick brown fox jumps over the lazy dog"
    sample_text = "if we wanted to make our watermark a simulation for visualizing could we code this up for generation"
    visualize_watermark_process(sample_text, watermark)

