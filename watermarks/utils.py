"""
Utility functions for watermarking, including PRNGs and helper functions.
"""

import hashlib
from typing import Optional


def position_hash(key: str, position: int, domain: str = "default") -> int:
    hash_input = f"{key}:{domain}:{position}".encode('utf-8')
    hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
    return hash_value


def get_binary_value(
    key: str, 
    position: int, 
    percentage: float = 0.5,
    domain: str = "binary"
) -> int:
    if not 0.0 <= percentage <= 1.0:
        raise ValueError(f"percentage must be between 0.0 and 1.0, got {percentage}")
    
    hash_value = position_hash(key, position, domain)
    
    threshold = int(percentage * (2**256 - 1))
    
    return 1 if hash_value <= threshold else 0


def get_green_positions(
    key: str,
    response_length: int,
    green_percentage: float = 0.5,
    start_position: int = 0
) -> set:
    green_positions = set()
    
    for pos in range(start_position, start_position + response_length):
        if get_binary_value(key, pos, green_percentage) == 1:
            green_positions.add(pos)
    
    return green_positions


def is_position_green(
    key: str,
    position: int,
    green_percentage: float = 0.5
) -> bool:
    return get_binary_value(key, position, green_percentage) == 1

