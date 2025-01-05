def test_constructor(generator):
    """Test the constructor implementation"""
    try:
        
        # Test client initialization
        assert hasattr(generator, 'client'), "OpenAI client not initialized"
        
        # Test prompt templates
        assert hasattr(generator, 'prompt_templates'), "Prompt templates not initialized"
        assert len(generator.prompt_templates) == 3, "Missing prompt templates"
        assert all(style in generator.prompt_templates for style in ['basic', 'detailed', 'artistic']), \
            "Missing required style templates"
        
        # Test template content structure
        for style in ['basic', 'detailed', 'artistic']:
            assert isinstance(generator.prompt_templates[style], str), \
                f"Template for {style} style is not a string"
            assert "{emoji}" in generator.prompt_templates[style], \
                f"Template for {style} style missing {{emoji}} placeholder"
        
        print("✓ Constructor test passed!")
        return True
    except AssertionError as e:
        print(f"✗ Constructor test failed: {str(e)}")
        return False

def test_emoji_validation(generator):
    """Test the emoji validation implementation"""
    try:
        
        # Test valid emoji inputs
        assert generator.is_valid_emoji("🌞") == True, "Single emoji not recognized"
        assert generator.is_valid_emoji("🌞🌊") == True, "Multiple emojis not recognized"
        assert generator.is_valid_emoji("Hello 🌞") == True, "Emoji with text not recognized"
        
        # Test invalid inputs
        assert generator.is_valid_emoji("hello") == False, "Text incorrectly recognized as emoji"
        assert generator.is_valid_emoji("") == False, "Empty string incorrectly recognized as emoji"
        assert generator.is_valid_emoji(":-)") == False, "Emoticon incorrectly recognized as emoji"
        assert generator.is_valid_emoji("123") == False, "Numbers incorrectly recognized as emoji"
        
        print("✓ Emoji validation test passed!")
        return True
    except AssertionError as e:
        print(f"✗ Emoji validation test failed: {str(e)}")
        return False


def run_all_tests(generator):
    """Run all tests on the student's implementation"""
    print("\n=== Running Tests ===\n")
    
    results = {
        "Constructor": test_constructor(generator),
        "Emoji Validation": test_emoji_validation(generator),
    }
    
    # Print summary
    print("\n=== Test Summary ===")
    total_passed = sum(results.values())
    for test, passed in results.items():
        status = "✓ Passed" if passed else "✗ Failed"
        print(f"{test}: {status}")
    print(f"\nTotal Passed: {total_passed}/{len(results)}")
    
    return total_passed == len(results)