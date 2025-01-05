def print_menu():
    """Print the main menu"""
    print("\n=== Emoji to Image Creator ===")
    print("1. Generate image")
    print("2. Change style")
    print("3. Exit")
    return input("Select an option (1-3): ")

def get_style():
    """Get style selection from user"""
    print("\nAvailable styles:")
    print("1. Basic - Simple representation")
    print("2. Detailed - Realistic scene")
    print("3. Artistic - Creative interpretation")
    
    style_map = {"1": "basic", "2": "detailed", "3": "artistic"}
    choice = input("Select style (1-3): ")
    return style_map.get(choice, "basic")
