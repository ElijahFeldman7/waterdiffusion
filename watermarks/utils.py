"""
Utility functions for watermarking, including PRNGs and helper functions.
"""

import hashlib
from typing import Optional


def position_hash(key: str, position: int, domain: str = "default") -> int:
    """
    Generate a hash value for a given key, position, and domain.
    
    Args:
        key: Secret key for watermarking
        position: Position index in the sequence
        domain: Domain string for hash separation (default: "default")
        
    Returns:
        Integer hash value
    """
    hash_input = f"{key}:{domain}:{position}".encode('utf-8')
    hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
    return hash_value


def get_binary_value(
    key: str, 
    position: int, 
    percentage: float = 0.5,
    domain: str = "binary"
) -> int:
    """
    Generate a deterministic binary value (0 or 1) for a given key and position.
    
    Uses the key and position to generate a deterministic hash, then returns
    0 or 1 based on whether the hash falls within the specified percentage.
    
    Args:
        key: Secret key for watermarking
        position: Position index in the sequence (1 to L)
        percentage: Percentage of positions that should return 1 (0.0 to 1.0)
        domain: Domain string for hash separation (default: "binary")
        
    Returns:
        0 or 1
        
    Example:
        >>> get_binary_value("my_key", 5, percentage=0.3)
        1  # Will be consistent for same key/position
        >>> get_binary_value("my_key", 5, percentage=0.3)
        1  # Same result
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError(f"percentage must be between 0.0 and 1.0, got {percentage}")
    
    hash_value = position_hash(key, position, domain)
    
    threshold = int(percentage * (2**256 - 1))
    
    return 1 if hash_value <= threshold else 0


def get_green_positions(
    key: str,
    response_length: int,
    green_percentage: float = 0.5,
    start_position: int = 1
) -> set:
    """
    Get all positions that should be marked as "green" (watermarked).
    
    Args:
        key: Secret key for watermarking
        response_length: Total length of the response (L)
        green_percentage: Percentage of positions that should be green (0.0 to 1.0)
        start_position: Starting position index (default: 1)
        
    Returns:
        Set of position indices that are green
        
    Example:
        >>> green_pos = get_green_positions("key", 10, 0.3)
        >>> len(green_pos)  # Approximately 3 positions
        3
    """
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
    """
    Check if a specific position is green (watermarked).
    
    Args:
        key: Secret key for watermarking
        position: Position index to check
        green_percentage: Percentage of positions that should be green
        
    Returns:
        True if position is green, False otherwise
    """
    return get_binary_value(key, position, green_percentage) == 1

