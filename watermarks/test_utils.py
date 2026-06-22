from watermarks.utils import get_binary_value, get_green_positions, is_position_green


def test_binary_value():
    
    key = "test_secret_key"
    
    print("\n1. Testing with 50% green rate:")
    results = []
    for pos in range(0, 10):
        value = get_binary_value(key, pos, percentage=0.5)
        results.append(value)
        print(f"  Position {pos}: {value}")
    print(f"  Green count: {sum(results)}/10 (expected ~5)")
    
    print("\n2. Testing with 30% green rate:")
    results = []
    for pos in range(0, 10):
        value = get_binary_value(key, pos, percentage=0.3)
        results.append(value)
        print(f"  Position {pos}: {value}")
    print(f"  Green count: {sum(results)}/10 (expected ~3)")
    
    print("\n3. Testing determinism (same key/position should give same result):")
    pos = 5
    val1 = get_binary_value(key, pos, percentage=0.5)
    val2 = get_binary_value(key, pos, percentage=0.5)
    print(f"  First call:  {val1}")
    print(f"  Second call: {val2}")
    print(f"  Deterministic: {val1 == val2}")
    
    print("\n4. Testing different keys give different results:")
    key1 = "key1"
    key2 = "key2"
    val1 = get_binary_value(key1, 1, percentage=0.5)
    val2 = get_binary_value(key2, 1, percentage=0.5)
    print(f"  Key1 at pos 1: {val1}")
    print(f"  Key2 at pos 1: {val2}")


def test_green_positions():
    """
    Test the get_green_positions function.
    """
    print("\n\nTesting get_green_positions...")
    
    key = "watermark_key_123"
    response_length = 20
    
    print(f"\nKey: {key}")
    print(f"Response length: {response_length}")
    
    for percentage in [0.2, 0.5, 0.8]:
        green_pos = get_green_positions(key, response_length, percentage)
        print(f"\nGreen percentage: {percentage*100}%")
        print(f"Green positions: {sorted(green_pos)}")
        print(f"Actual green count: {len(green_pos)}/{response_length} = {len(green_pos)/response_length:.2%}")


def test_is_position_green():
    """
    Test the is_position_green function.
    """
    print("\n\nTesting is_position_green...")
    
    key = "my_secret"
    green_percentage = 0.4
    
    print(f"\nKey: {key}")
    print(f"Green percentage: {green_percentage*100}%")
    print("\nChecking positions 0-14:")
    
    for pos in range(0, 15):
        is_green = is_position_green(key, pos, green_percentage)

