import hashlib
import random
from typing import Optional

def generate_token_pairs(key: str, vocab_size: int, mask_token: Optional[int] = None) -> set:
    seed = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    valid_tokens = [i for i in range(vocab_size) if i != mask_token]
    pairs = set()
    
    for i in valid_tokens:
        other_tokens = [j for j in valid_tokens if j != i]
        rng.shuffle(other_tokens)
        half_point = len(other_tokens) // 2
        for j in other_tokens[:half_point]:
            if (i, j) not in pairs and (j, i) not in pairs: pairs.add((j, i))

        for j in other_tokens[half_point:]:
            if (i, j) not in pairs and (j, i) not in pairs: pairs.add((i, j))
    
    return pairs


