from typing import Dict, List
import json

class TestAIWritingAssistant:
    """Test methods for AI Writing Assistant implementation"""
    
    def __init__(self, assistant):
        """Initialize test suite"""
        self.assistant = assistant
        self.results = {
            "passed": 0,
            "failed": 0,
            "total": 0
        }
        
        # Test data
        self.test_data = {
            "topic": "Python Programming",
            "content_type": "tutorial",
            "audience": "beginners",
            "content": "Python is a programming language.",
            "examples": [
                "Example 1: Python is a high-level programming language known for its simplicity and readability. It emphasizes clean syntax and extensive library support, making it ideal for beginners. The language supports multiple programming paradigms and has a large, active community.",
                "Example 2: JavaScript is a versatile programming language primarily used for web development. It enables interactive web pages and is essential for modern web applications. With its event-driven architecture and extensive framework ecosystem, it's a crucial tool for web developers."
            ]
        }
    
    def run_test(self, test_name: str, test_function) -> bool:
        """Run a single test and record result"""
        self.results["total"] += 1
        try:
            test_function()
            self.results["passed"] += 1
            print(f"✅ {test_name}: Passed")
            return True
        except Exception as e:
            self.results["failed"] += 1
            print(f"❌ {test_name}: Failed - {str(e)}")
            return False
    
    def test_initial_content_generation(self) -> bool:
        """Test zero-shot content generation"""
        content = self.assistant.generate_initial_content(
            self.test_data["topic"],
            self.test_data["content_type"],
            self.test_data["audience"]
        )
        
        # Verify content quality
        if not content:
            raise Exception("No content generated")
        if len(content) < 200:
            raise Exception("Content too short")
        if self.test_data["topic"].lower() not in content.lower():
            raise Exception("Topic not found in content")
            
        return True
    
    def test_content_improvement(self) -> bool:
        """Test few-shot learning implementation"""
        improved = self.assistant.improve_content_with_examples(
            self.test_data["content"],
            self.test_data["examples"]
        )
        
        # Verify improvements
        if not improved:
            raise Exception("No improved content generated")
        if len(improved) <= len(self.test_data["content"]):
            raise Exception("Content not improved")
        if improved == self.test_data["content"]:
            raise Exception("Content unchanged")
            
        return True
    
    def test_content_analysis(self) -> bool:
        """Test chain-of-thought analysis"""
        analysis = self.assistant.structured_content_analysis(
            self.test_data["content"]
        )
        
        # Check analysis structure
        required_keys = {
            'structure_score',
            'clarity_score',
            'tone_assessment',
            'improvement_areas',
            'specific_recommendations'
        }
        
        if not isinstance(analysis, dict):
            raise Exception("Analysis not in dictionary format")
        if not all(key in analysis for key in required_keys):
            raise Exception("Missing required analysis keys")
        if not 1 <= analysis['structure_score'] <= 10:
            raise Exception("Invalid structure score")
        if not 1 <= analysis['clarity_score'] <= 10:
            raise Exception("Invalid clarity score")
            
        return True
    
    def test_content_revision(self) -> bool:
        """Test plan-and-solve implementation"""
        test_analysis = {
            'structure_score': 7,
            'clarity_score': 6,
            'tone_assessment': 'Professional',
            'improvement_areas': ['Add more examples'],
            'specific_recommendations': ['Include code snippets']
        }
        
        revised = self.assistant.plan_and_revise(
            self.test_data["content"],
            test_analysis
        )
        
        if not revised:
            raise Exception("No revised content generated")
        if len(revised) <= len(self.test_data["content"]):
            raise Exception("Content not revised")
        if revised == self.test_data["content"]:
            raise Exception("Content unchanged")
            
        return True
    
    def test_complete_workflow(self) -> bool:
        """Test entire content creation process"""
        result = self.assistant.create_polished_content(
            self.test_data["topic"],
            self.test_data["content_type"],
            self.test_data["audience"],
            self.test_data["examples"]
        )
        
        required_keys = {
            'initial_content',
            'improved_content',
            'analysis',
            'final_content'
        }
        
        if not all(key in result for key in required_keys):
            raise Exception("Missing required result keys")
        if result['initial_content'] == result['final_content']:
            raise Exception("Content not evolved")
        if not isinstance(result['analysis'], dict):
            raise Exception("Invalid analysis format")
            
        return True
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return results"""
        print("\n🏁 Starting AI Writing Assistant Tests...\n")
        
        # Run all tests
        self.run_test("Initial Content Generation", self.test_initial_content_generation)
        self.run_test("Content Improvement", self.test_content_improvement)
        self.run_test("Content Analysis", self.test_content_analysis)
        self.run_test("Content Revision", self.test_content_revision)
        self.run_test("Complete Workflow", self.test_complete_workflow)        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"Total Tests: {self.results['total']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        return self.results
