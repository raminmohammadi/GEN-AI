# Dataclass for email analysis results 
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EmailAnalysis:
    category: str
    priority: str
    deadlines: List[str]
    requests: List[str]
    questions: List[str]
    action_items: List[str]
    sentiment: str

# Dataclass for thread summary results
@dataclass
class ThreadSummary:
    key_points: List[str]
    decisions: List[str]
    action_items: List[str]
    participants: List[str]
    timeline: List[Dict[str, str]]
    
    
class TestEmailAssistant:
    def __init__(self, email_assistant):
        """
        Initialize test class with an EmailAssistant instance
        
        Args:
            email_assistant: An instance of EmailAssistant class to be tested
        """
            
        self.assistant = email_assistant
        
        # Sample test data
        self.test_email = """
        Dear Team,
        
        I hope you're doing well. We need to submit the project proposal by next Friday.
        Could you please review the following points:
        
        1. Budget estimation
        2. Timeline details
        
        Also, when can we schedule the team meeting?
        
        Best regards,
        Alice
        """
        
        self.test_thread = [
            {
                "from": "Alice",
                "date": "2023-12-01",
                "content": self.test_email
            },
            {
                "from": "Bob",
                "date": "2023-12-02",
                "content": "I can review the budget by Wednesday. Available for meeting on Thursday."
            }
        ]
        
        # Track test results
        self.tests_passed = 0
        self.tests_failed = 0

    def print_result(self, test_name, passed, error=None):
        """Helper method to print test results"""
        if passed:
            print(f"✅ {test_name} passed")
            self.tests_passed += 1
        else:
            print(f"❌ {test_name} failed" + (f": {error}" if error else ""))
            self.tests_failed += 1

    def test_implementation(self):
        """Test if EmailAssistant has required attributes and methods"""
        try:
            # Check required attributes
            required_attrs = ['client', 'model', 'cache']
            for attr in required_attrs:
                if not hasattr(self.assistant, attr):
                    raise ValueError(f"Missing required attribute: {attr}")
                    
            # Check required methods
            required_methods = [
                '_get_completion',
                'analyze_email',
                'generate_response',
                'summarize_thread'
            ]
            for method in required_methods:
                if not hasattr(self.assistant, method):
                    raise ValueError(f"Missing required method: {method}")
                    
            self.print_result("Implementation check", True)
        except Exception as e:
            self.print_result("Implementation check", False, str(e))

    def test_completion(self):
        """Test the _get_completion method"""
        try:
            messages = [{"role": "user", "content": "test"}]
            response = self.assistant._get_completion(messages)
            
            if not isinstance(response, str):
                raise ValueError("Response should be a string")
                
            self.print_result("Completion test", True)
        except Exception as e:
            self.print_result("Completion test", False, str(e))

    def test_email_analysis(self):
        """Test email analysis functionality"""
        try:
            analysis = self.assistant.analyze_email(self.test_email)
                
            # Check required attributes
            required_attrs = ['category', 'priority', 'deadlines', 'requests', 
                            'questions', 'action_items', 'sentiment']
            for attr in required_attrs:
                if not hasattr(analysis, attr):
                    raise ValueError(f"Missing attribute: {attr}")
                    
            self.print_result("Email analysis test", True)
        except Exception as e:
            self.print_result("Email analysis test", False, str(e))

    def test_response_generation(self):
        """Test response generation functionality"""
        try:
            analysis = EmailAnalysis(
                category="important",
                priority="high",
                deadlines=["next Friday"],
                requests=["review budget"],
                questions=["when can we meet?"],
                action_items=["review proposal"],
                sentiment="neutral"
            )
            
            response = self.assistant.generate_response(self.test_email, analysis)
            
            if not isinstance(response, str) or len(response) == 0:
                raise ValueError("Invalid response generated")
                
            self.print_result("Response generation test", True)
        except Exception as e:
            self.print_result("Response generation test", False, str(e))

    def test_thread_summarization(self):
        """Test thread summarization functionality"""
        try:
            summary = self.assistant.summarize_thread(self.test_thread)
                
            # Check required attributes
            required_attrs = ['key_points', 'decisions', 'action_items', 
                            'participants', 'timeline']
            for attr in required_attrs:
                if not hasattr(summary, attr):
                    raise ValueError(f"Missing attribute: {attr}")
                    
            self.print_result("Thread summarization test", True)
        except Exception as e:
            self.print_result("Thread summarization test", False, str(e))

    def test_error_handling(self):
        """Test error handling in various methods"""
        try:
            # Test with empty email
            try:
                self.assistant.analyze_email("")
                raise Exception("Should have raised error for empty email")
            except:
                pass
                
            # Test with invalid thread
            try:
                self.assistant.summarize_thread([])
                raise Exception("Should have raised error for empty thread")
            except:
                pass
                
            self.print_result("Error handling test", True)
        except Exception as e:
            self.print_result("Error handling test", False, str(e))

    def run_all_tests(self):
        """Run all tests and print summary"""
        print("\nTesting EmailAssistant Implementation...")
        print("-" * 40)
        
        self.test_implementation()
        self.test_completion()
        self.test_email_analysis()
        self.test_response_generation()
        self.test_thread_summarization()
        self.test_error_handling()
        
        print("-" * 40)
        print(f"\nTest Summary:")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ {self.tests_failed} test(s) failed!")
