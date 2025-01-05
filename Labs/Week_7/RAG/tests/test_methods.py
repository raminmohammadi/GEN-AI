import unittest
from typing import Dict, List, Union
import numpy as np
import json
import time

class TestRAGSystem:
    def __init__(self, rag_system):
        """Initialize the test suite with a RAG system instance"""
        self.rag_system = rag_system
        self.test_documents = [
            "The quick brown fox jumps over the lazy dog. This is a test document that contains some sample text.",
            "Artificial Intelligence is transforming various industries. Machine learning models can identify patterns.",
            "Climate change is a global challenge. Renewable energy sources are becoming increasingly important.",
            "The human brain processes information through neural networks. Neuroscience research continues to advance."
        ]
        self.test_formats = ["text"] * len(self.test_documents)
        self.test_queries = [
            "What can you tell me about artificial intelligence?",
            "How does the human brain process information?",
            "What is the significance of renewable energy?",
            "Tell me about pattern recognition in machine learning."
        ]

    def test_document_processing(self) -> Dict:
        """Test document processing functionality"""
        try:
            start_time = time.time()
            
            # Test document chunking
            chunks = self.rag_system.doc_processor.process_document(
                self.test_documents[0], "text"
            )
            
            # Validate chunk structure
            assert isinstance(chunks, list), "Chunks should be a list"
            assert all(isinstance(chunk, dict) for chunk in chunks), "Each chunk should be a dictionary"
            assert all("text" in chunk for chunk in chunks), "Each chunk should contain text"
            assert all("metadata" in chunk for chunk in chunks), "Each chunk should contain metadata"
            
            # Test overlap
            if len(chunks) > 1:
                text1 = chunks[0]["text"]
                text2 = chunks[1]["text"]
                assert any(word in text2 for word in text1.split()), "Chunks should have overlap"
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "chunks_generated": len(chunks)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "document_processing"
            }

    def test_embedding_generation(self) -> Dict:
        """Test embedding generation functionality"""
        try:
            start_time = time.time()
            
            # Generate embeddings for test text
            test_text = ["This is a test sentence for embedding generation."]
            embeddings = self.rag_system.embedding_generator.generate_embeddings(test_text)
            
            # Validate embeddings
            assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
            assert embeddings.shape[1] == 1536, "OpenAI embeddings should be 1536-dimensional"
            assert not np.any(np.isnan(embeddings)), "Embeddings should not contain NaN values"
            
            # Test caching
            cache_start = time.time()
            cached_embeddings = self.rag_system.embedding_generator.generate_embeddings(test_text)
            cache_time = time.time() - cache_start
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "cache_time": cache_time,
                "embedding_shape": embeddings.shape
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "embedding_generation"
            }

    def test_vector_store(self) -> Dict:
        """Test vector store functionality"""
        try:
            start_time = time.time()
            
            # Test document addition
            test_embeddings = np.random.rand(2, 1536)
            test_docs = [
                {"text": "Test document 1", "metadata": {"source": "test1"}},
                {"text": "Test document 2", "metadata": {"source": "test2"}}
            ]
            
            self.rag_system.vector_store.add_documents(test_docs, test_embeddings)
            
            # Test similarity search
            query_embedding = np.random.rand(1, 1536)
            results = self.rag_system.vector_store.similarity_search(query_embedding)
            
            assert isinstance(results, list), "Search results should be a list"
            assert len(results) > 0, "Search should return results"
            assert all("text" in doc for doc in results), "Results should contain text"
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "results_returned": len(results)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "vector_store"
            }

    def test_context_building(self) -> Dict:
        """Test context building functionality"""
        try:
            start_time = time.time()
            
            # Test context building with sample documents
            relevant_docs = [
                {"text": "Sample text 1", "score": 0.9},
                {"text": "Sample text 2", "score": 0.8}
            ]
            query = "Test query"
            
            context = self.rag_system.context_builder.build_context(relevant_docs, query)
            
            assert isinstance(context, str), "Context should be a string"
            assert len(context) > 0, "Context should not be empty"
            
            # Test token limit
            token_count = len(context.split())
            assert token_count <= self.rag_system.context_builder.max_tokens, "Context should respect token limit"
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "context_length": len(context),
                "token_count": token_count
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "context_building"
            }

    def test_response_generation(self) -> Dict:
        """Test response generation functionality"""
        try:
            start_time = time.time()
            
            # Test response generation
            query = "What is artificial intelligence?"
            context = "Artificial Intelligence is a technology that enables computers to simulate human intelligence."
            
            response = self.rag_system.response_generator.generate_response(query, context)
            
            assert isinstance(response, dict), "Response should be a dictionary"
            assert "answer" in response, "Response should contain an answer"
            assert "sources" in response, "Response should contain sources"
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "response_length": len(response["answer"])
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "response_generation"
            }

    def test_end_to_end(self) -> Dict:
        """Test complete RAG pipeline"""
        try:
            start_time = time.time()
            
            # Add test documents
            self.rag_system.add_documents(self.test_documents, self.test_formats)
            
            # Test querying
            query = self.test_queries[0]
            result = self.rag_system.query(query)
            
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "answer" in result, "Result should contain an answer"
            assert "sources" in result, "Result should contain sources"
            
            return {
                "status": "passed",
                "time_taken": time.time() - start_time,
                "documents_processed": len(self.test_documents),
                "response_received": bool(result)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "component": "end_to_end"
            }

    def run_all_tests(self) -> Dict:
        """Run all test cases and return results"""
        results = {
            "document_processing": self.test_document_processing(),
            "embedding_generation": self.test_embedding_generation(),
            "vector_store": self.test_vector_store(),
            "context_building": self.test_context_building(),
            "response_generation": self.test_response_generation(),
            "end_to_end": self.test_end_to_end()
        }
        
        # Calculate overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for test in results.values() if test["status"] == "passed")
        total_time = sum(test["time_taken"] for test in results.values())
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_time": total_time,
            "success_rate": (passed_tests / total_tests) * 100
        }
        
        results["summary"] = summary
        
        # Print results in a readable format
        print("\n=== RAG System Test Results ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Total Time: {summary['total_time']:.2f} seconds")
        print("\nDetailed Results:")
        for component, result in results.items():
            if component != "summary":
                status = result["status"]
                print(f"\n{component}: {status.upper()}")
                if status == "failed":
                    print(f"Error: {result['error']}")
        
        return results