import hashlib

def pair_rule_follows(key: str, token_a: int, token_b: int) -> bool:
    a = int(token_a)
    b = int(token_b)
    if a == b:
        return False
    lo, hi = (a, b) if a < b else (b, a)
    h = hashlib.sha256()
    h.update(key.encode('utf-8'))
    h.update(b'|')
    h.update(str(lo).encode('utf-8'))
    h.update(b'|')
    h.update(str(hi).encode('utf-8'))
    v = int.from_bytes(h.digest()[:2], 'big')
    bit = v & 1
    if a < b:
        return bit == 1
    else:
        return bit == 0


