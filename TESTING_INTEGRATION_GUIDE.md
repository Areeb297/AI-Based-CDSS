# CDSS Testing and Integration Guide

## Overview

This guide provides comprehensive testing strategies, integration patterns, and validation approaches for the Clinical Decision Support System (CDSS). It includes unit tests, integration tests, performance tests, and real-world usage examples.

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [API Testing](#api-testing)
5. [Performance Testing](#performance-testing)
6. [Validation Testing](#validation-testing)
7. [Integration Patterns](#integration-patterns)
8. [Deployment Testing](#deployment-testing)
9. [Monitoring and Logging](#monitoring-and-logging)

## Testing Strategy

### Testing Pyramid

```
    ┌─────────────────┐
    │   E2E Tests     │  ← Full workflow validation
    │                 │
    ├─────────────────┤
    │ Integration     │  ← API endpoint testing
    │ Tests           │
    ├─────────────────┤
    │   Unit Tests    │  ← Individual function testing
    │                 │
    └─────────────────┘
```

### Test Categories

1. **Unit Tests**: Individual function validation
2. **Integration Tests**: Component interaction testing
3. **API Tests**: Endpoint functionality validation
4. **Performance Tests**: Load and response time testing
5. **Validation Tests**: Clinical accuracy verification

## Unit Testing

### Test Setup

```python
# test_cdss_functions.py
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from cdss_app import (
    extract_json_from_response,
    enforce_contract,
    validate_clinical_request,
    ClinicalRequest
)

class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_extract_json_from_response_with_code_block(self):
        """Test JSON extraction from code block format."""
        response_text = """
        Here's the result:
        ```json
        {"key": "value", "number": 42}
        ```
        """
        expected = {"key": "value", "number": 42}
        result = extract_json_from_response(response_text)
        assert result == expected
    
    def test_extract_json_from_response_inline(self):
        """Test JSON extraction from inline format."""
        response_text = 'Some text {"status": "success"} more text'
        expected = {"status": "success"}
        result = extract_json_from_response(response_text)
        assert result == expected
    
    def test_extract_json_from_response_invalid(self):
        """Test JSON extraction with invalid input."""
        response_text = "No JSON here!"
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json_from_response(response_text)
    
    def test_extract_json_malformed(self):
        """Test JSON extraction with malformed JSON."""
        response_text = '{"key": "value"'  # Missing closing brace
        with pytest.raises(ValueError, match="Failed to decode JSON"):
            extract_json_from_response(response_text)

class TestContractEnforcement:
    """Test suite for contract enforcement functions."""
    
    def test_enforce_contract_complete(self):
        """Test contract enforcement with complete data."""
        output_json = {
            "Drug-Drug Interactions": ["Risk of bleeding [1][2]"],
            "Drug-Allergy": ["No known allergies [3]"],
            "Severity": ["High"]
        }
        references_dict = {
            "[1]": "FDA label for warfarin",
            "[2]": "FDA label for ibuprofen", 
            "[3]": "Allergy database"
        }
        expected_keys = ["Drug-Drug Interactions", "Drug-Allergy", "Severity", "References"]
        
        result = enforce_contract(output_json, references_dict, expected_keys)
        
        assert "References" in result
        assert len(result["References"]) == 3
        assert "[1] FDA label for warfarin" in result["References"]
    
    def test_enforce_contract_missing_keys(self):
        """Test contract enforcement with missing keys."""
        output_json = {"Drug-Drug Interactions": ["Some interaction [1]"]}
        references_dict = {"[1]": "Reference text"}
        expected_keys = ["Drug-Drug Interactions", "Drug-Allergy", "References"]
        
        result = enforce_contract(output_json, references_dict, expected_keys)
        
        assert result["Drug-Allergy"] == ["None"]
        assert "References" in result

class TestValidation:
    """Test suite for validation functions."""
    
    def test_validate_clinical_request_valid(self):
        """Test validation with valid clinical request."""
        request = ClinicalRequest(
            drugs=["aspirin", "ibuprofen"],
            age=45,
            gender="Female",
            allergies=["penicillin"],
            diagnosis="Hypertension"
        )
        assert validate_clinical_request(request) == True
    
    def test_validate_clinical_request_no_drugs(self):
        """Test validation with no drugs specified."""
        request = ClinicalRequest(
            drugs=[],
            age=45,
            gender="Female"
        )
        with pytest.raises(ValueError, match="At least one drug must be specified"):
            validate_clinical_request(request)
    
    def test_validate_clinical_request_invalid_age(self):
        """Test validation with invalid age."""
        request = ClinicalRequest(
            drugs=["aspirin"],
            age=200,
            gender="Female"
        )
        with pytest.raises(ValueError, match="Age must be between 0 and 150"):
            validate_clinical_request(request)
    
    def test_validate_clinical_request_invalid_gender(self):
        """Test validation with invalid gender."""
        request = ClinicalRequest(
            drugs=["aspirin"],
            age=45,
            gender="Other"
        )
        with pytest.raises(ValueError, match="Gender must be 'Male' or 'Female'"):
            validate_clinical_request(request)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### RAG Pipeline Tests

```python
# test_rag_pipeline.py
import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document
from cdss_app import filter_document_chunks, create_or_load_vector_store

class TestRAGPipeline:
    """Test suite for RAG pipeline components."""
    
    def test_filter_document_chunks_valid(self):
        """Test filtering with valid document chunks."""
        chunks = [
            Document(page_content="Valid content 1", metadata={"page": 1}),
            Document(page_content="Valid content 2", metadata={"page": 2}),
            Document(page_content="", metadata={"page": 3}),  # Empty content
            Document(page_content="Valid content 3", metadata={"page": 4})
        ]
        
        result = filter_document_chunks(chunks)
        
        assert len(result) == 3
        assert all(chunk.page_content.strip() for chunk in result)
    
    def test_filter_document_chunks_empty(self):
        """Test filtering with no valid chunks."""
        chunks = [
            Document(page_content="", metadata={"page": 1}),
            Document(page_content="   ", metadata={"page": 2})
        ]
        
        with pytest.raises(ValueError, match="No valid document chunks found"):
            filter_document_chunks(chunks)
    
    @patch('cdss_app.FAISS')
    @patch('cdss_app.Path')
    def test_create_vector_store_existing(self, mock_path, mock_faiss):
        """Test vector store creation with existing index."""
        # Mock existing files
        mock_path.return_value.__truediv__.return_value.exists.return_value = True
        mock_faiss.load_local.return_value = Mock()
        
        documents = [Document(page_content="Test content")]
        embeddings = Mock()
        
        result = create_or_load_vector_store("test_path", documents, embeddings)
        
        mock_faiss.load_local.assert_called_once()
        assert result is not None
    
    @patch('cdss_app.FAISS')
    @patch('cdss_app.Path')
    def test_create_vector_store_new(self, mock_path, mock_faiss):
        """Test vector store creation without existing index."""
        # Mock non-existing files
        mock_path.return_value.__truediv__.return_value.exists.return_value = False
        mock_vectordb = Mock()
        mock_faiss.from_documents.return_value = mock_vectordb
        
        documents = [Document(page_content="Test content")]
        embeddings = Mock()
        
        result = create_or_load_vector_store("test_path", documents, embeddings)
        
        mock_faiss.from_documents.assert_called_once_with(documents, embeddings)
        mock_vectordb.save_local.assert_called_once_with("test_path")
```

## Integration Testing

### API Integration Tests

```python
# test_api_integration.py
import pytest
import requests
import json
from fastapi.testclient import TestClient
from cdss_app import app

class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        self.base_url = "http://testserver"
    
    def test_clinical_assess_endpoint(self):
        """Test clinical assessment endpoint integration."""
        request_data = {
            "drugs": ["warfarin", "ibuprofen"],
            "age": 70,
            "gender": "Male",
            "allergies": [],
            "diagnosis": "Atrial Fibrillation and Osteoarthritis"
        }
        
        response = self.client.post("/clinical-assess", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify expected keys are present
        expected_keys = [
            "Drug-Drug Interactions", "Drug-Allergy", "Drug-Disease Contraindications",
            "Ingredient Duplication", "Pregnancy Warnings", "Lactation Warnings",
            "General Precautions", "Therapeutic Class Conflicts", "Warning Labels",
            "Indications", "Severity", "References"
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Verify data types
        assert isinstance(result["Severity"], list)
        assert isinstance(result["References"], list)
    
    def test_clinical_pathways_endpoint(self):
        """Test clinical pathways endpoint integration."""
        request_data = {
            "input": {
                "drugs": ["lisinopril", "ibuprofen"],
                "age": 34,
                "gender": "Female",
                "allergies": ["sulfa drugs"],
                "diagnosis": "Gestational Hypertension"
            },
            "severity": "high"
        }
        
        response = self.client.post("/clinical-pathways", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify structure
        assert "clinical_pathways" in result
        assert len(result["clinical_pathways"]) == 3
        
        # Verify each pathway has required fields
        for pathway in result["clinical_pathways"]:
            assert "recommendation" in pathway
            assert "details" in pathway
            assert "tradeoff" in pathway
            assert "monitoring" in pathway
    
    def test_clinical_assess_invalid_input(self):
        """Test clinical assessment with invalid input."""
        request_data = {
            "drugs": [],  # Empty drugs list
            "age": 70,
            "gender": "Male"
        }
        
        response = self.client.post("/clinical-assess", json=request_data)
        
        # Should handle validation error gracefully
        assert response.status_code in [400, 422]
    
    @pytest.mark.slow
    def test_generate_report_endpoint(self):
        """Test radiology report generation endpoint."""
        # Note: This test requires actual image file
        # In practice, you'd use a test image
        test_image_path = "test_chest_xray.jpg"
        
        with open(test_image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {"query": "Test patient with chest pain"}
            
            response = self.client.post("/generate_report/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                assert "View" in result
                assert "Findings" in result
                assert "Impression" in result
                assert "Recommendations" in result
```

### End-to-End Testing

```python
# test_e2e_scenarios.py
import pytest
import requests
import time
from typing import List, Dict

class TestEndToEndScenarios:
    """End-to-end testing scenarios."""
    
    def setup_method(self):
        """Set up for E2E tests."""
        self.base_url = "http://localhost:8000"
        self.timeout = 60  # seconds
    
    def test_complete_clinical_workflow(self):
        """Test complete clinical decision workflow."""
        # Step 1: Clinical Assessment
        assessment_request = {
            "drugs": ["metformin", "lisinopril", "ibuprofen"],
            "age": 50,
            "gender": "Female",
            "allergies": ["penicillin"],
            "diagnosis": "Type 2 Diabetes Mellitus, Hypertension, Osteoarthritis"
        }
        
        assessment_response = requests.post(
            f"{self.base_url}/clinical-assess",
            json=assessment_request,
            timeout=self.timeout
        )
        
        assert assessment_response.status_code == 200
        assessment_result = assessment_response.json()
        
        # Verify high-risk interactions detected
        assert any("interaction" in interaction.lower() 
                  for interaction in assessment_result["Drug-Drug Interactions"])
        
        # Step 2: Generate Clinical Pathways
        pathways_request = {
            "input": assessment_request,
            "severity": "high"
        }
        
        pathways_response = requests.post(
            f"{self.base_url}/clinical-pathways",
            json=pathways_request,
            timeout=self.timeout
        )
        
        assert pathways_response.status_code == 200
        pathways_result = pathways_response.json()
        
        # Verify pathways provide alternatives
        pathways = pathways_result["clinical_pathways"]
        assert len(pathways) == 3
        
        # Low-risk option should avoid problematic combinations
        low_risk_pathway = pathways[0]
        assert "avoid" in low_risk_pathway["recommendation"].lower() or \
               "acetaminophen" in low_risk_pathway["recommendation"].lower()
        
        # High-risk option should include warning
        high_risk_pathway = pathways[2]
        assert "do not use" in high_risk_pathway["recommendation"].lower()
    
    def test_pregnancy_scenario(self):
        """Test pregnancy-specific clinical scenario."""
        request_data = {
            "drugs": ["lisinopril", "ibuprofen"],
            "age": 28,
            "gender": "Female",
            "allergies": [],
            "diagnosis": "Gestational Hypertension"
        }
        
        response = requests.post(
            f"{self.base_url}/clinical-assess",
            json=request_data,
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify pregnancy warnings are present
        pregnancy_warnings = result["Pregnancy Warnings"]
        assert pregnancy_warnings != ["None"]
        assert any("pregnancy" in warning.lower() for warning in pregnancy_warnings)
        
        # Verify high severity due to pregnancy risks
        assert result["Severity"] == ["High"]
    
    def test_pediatric_scenario(self):
        """Test pediatric-specific clinical scenario."""
        request_data = {
            "drugs": ["aspirin"],
            "age": 10,
            "gender": "Male",
            "allergies": [],
            "diagnosis": "Varicella (Chickenpox)"
        }
        
        response = requests.post(
            f"{self.base_url}/clinical-assess",
            json=request_data,
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify Reye's syndrome warning
        warnings = result["Warning Labels"]
        assert any("reye" in warning.lower() for warning in warnings)
        
        # Verify high severity
        assert result["Severity"] == ["High"]

def run_test_suite():
    """Run comprehensive test suite."""
    test_classes = [
        TestUtilityFunctions,
        TestContractEnforcement,
        TestValidation,
        TestRAGPipeline,
        TestAPIIntegration,
        TestEndToEndScenarios
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        pytest.main([f"test_{test_class.__name__.lower()}.py", "-v"])
```

## API Testing

### Postman Collection

```json
{
  "info": {
    "name": "CDSS API Tests",
    "description": "Comprehensive API testing collection for CDSS"
  },
  "item": [
    {
      "name": "Clinical Assessment - Valid Request",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"drugs\": [\"warfarin\", \"ibuprofen\"],\n  \"age\": 70,\n  \"gender\": \"Male\",\n  \"allergies\": [],\n  \"diagnosis\": \"Atrial Fibrillation and Osteoarthritis\"\n}"
        },
        "url": {
          "raw": "{{base_url}}/clinical-assess",
          "host": ["{{base_url}}"],
          "path": ["clinical-assess"]
        }
      },
      "response": []
    },
    {
      "name": "Clinical Pathways - High Risk",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"input\": {\n    \"drugs\": [\"warfarin\", \"ibuprofen\"],\n    \"age\": 70,\n    \"gender\": \"Male\",\n    \"allergies\": [],\n    \"diagnosis\": \"Atrial Fibrillation and Osteoarthritis\"\n  },\n  \"severity\": \"high\"\n}"
        },
        "url": {
          "raw": "{{base_url}}/clinical-pathways",
          "host": ["{{base_url}}"],
          "path": ["clinical-pathways"]
        }
      },
      "response": []
    },
    {
      "name": "Radiology Report - Chest X-ray",
      "request": {
        "method": "POST",
        "header": [],
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "image",
              "type": "file",
              "src": "chest_xray_sample.jpg"
            },
            {
              "key": "query",
              "value": "65-year-old patient with chest pain",
              "type": "text"
            }
          ]
        },
        "url": {
          "raw": "{{base_url}}/generate_report/",
          "host": ["{{base_url}}"],
          "path": ["generate_report", ""]
        }
      },
      "response": []
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

### API Test Scripts

```python
# test_api_comprehensive.py
import requests
import json
import time
from typing import Dict, List
import concurrent.futures

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_clinical_assess_scenarios(self) -> List[Dict]:
        """Test multiple clinical assessment scenarios."""
        test_cases = [
            {
                "name": "Warfarin-Ibuprofen Interaction",
                "data": {
                    "drugs": ["warfarin", "ibuprofen"],
                    "age": 70,
                    "gender": "Male",
                    "allergies": [],
                    "diagnosis": "Atrial Fibrillation and Osteoarthritis"
                },
                "expected_severity": "High"
            },
            {
                "name": "Pregnancy Scenario",
                "data": {
                    "drugs": ["lisinopril", "ibuprofen"],
                    "age": 28,
                    "gender": "Female",
                    "allergies": ["sulfa drugs"],
                    "diagnosis": "Gestational Hypertension"
                },
                "expected_severity": "High"
            },
            {
                "name": "Pediatric Aspirin",
                "data": {
                    "drugs": ["aspirin"],
                    "age": 10,
                    "gender": "Male",
                    "allergies": [],
                    "diagnosis": "Varicella (Chickenpox)"
                },
                "expected_severity": "High"
            },
            {
                "name": "Low Risk Combination",
                "data": {
                    "drugs": ["amlodipine", "acetaminophen"],
                    "age": 60,
                    "gender": "Male",
                    "allergies": ["aspirin"],
                    "diagnosis": "Primary Hypertension"
                },
                "expected_severity": "Low"
            }
        ]
        
        results = []
        for test_case in test_cases:
            print(f"Testing: {test_case['name']}")
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/clinical-assess",
                json=test_case["data"],
                timeout=60
            )
            end_time = time.time()
            
            result = {
                "test_name": test_case["name"],
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "passed": False,
                "details": {}
            }
            
            if response.status_code == 200:
                json_response = response.json()
                
                # Verify expected structure
                expected_keys = [
                    "Drug-Drug Interactions", "Drug-Allergy", "Severity", "References"
                ]
                
                structure_valid = all(key in json_response for key in expected_keys)
                severity_correct = json_response.get("Severity") == [test_case["expected_severity"]]
                
                result["passed"] = structure_valid and severity_correct
                result["details"] = {
                    "structure_valid": structure_valid,
                    "severity_correct": severity_correct,
                    "actual_severity": json_response.get("Severity"),
                    "reference_count": len(json_response.get("References", []))
                }
            
            results.append(result)
            print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"  Response time: {result['response_time']:.2f}s")
        
        return results
    
    def test_concurrent_requests(self, num_requests: int = 5) -> Dict:
        """Test concurrent API requests."""
        request_data = {
            "drugs": ["metformin", "naproxen"],
            "age": 55,
            "gender": "Male",
            "allergies": ["penicillin"],
            "diagnosis": "Type 2 Diabetes and Osteoarthritis"
        }
        
        def make_request():
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/clinical-assess",
                json=request_data,
                timeout=60
            )
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        print(f"Testing {num_requests} concurrent requests...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        max_response_time = max(r["response_time"] for r in results)
        
        return {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests * 100,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "all_results": results
        }
    
    def test_error_handling(self) -> List[Dict]:
        """Test API error handling."""
        error_test_cases = [
            {
                "name": "Empty drugs list",
                "data": {
                    "drugs": [],
                    "age": 45,
                    "gender": "Female"
                },
                "expected_status": [400, 422]
            },
            {
                "name": "Invalid age",
                "data": {
                    "drugs": ["aspirin"],
                    "age": -5,
                    "gender": "Male"
                },
                "expected_status": [400, 422]
            },
            {
                "name": "Invalid gender",
                "data": {
                    "drugs": ["aspirin"],
                    "age": 45,
                    "gender": "Unknown"
                },
                "expected_status": [400, 422]
            },
            {
                "name": "Missing required fields",
                "data": {
                    "drugs": ["aspirin"]
                    # Missing age and gender
                },
                "expected_status": [400, 422]
            }
        ]
        
        results = []
        for test_case in error_test_cases:
            print(f"Testing error case: {test_case['name']}")
            
            response = requests.post(
                f"{self.base_url}/clinical-assess",
                json=test_case["data"],
                timeout=30
            )
            
            result = {
                "test_name": test_case["name"],
                "status_code": response.status_code,
                "expected_status": test_case["expected_status"],
                "passed": response.status_code in test_case["expected_status"]
            }
            
            results.append(result)
            print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"  Response code: {result['status_code']}")
        
        return results
    
    def run_full_test_suite(self) -> Dict:
        """Run complete API test suite."""
        print("=== CDSS API Test Suite ===\n")
        
        # Test 1: Clinical Assessment Scenarios
        print("1. Testing Clinical Assessment Scenarios...")
        assessment_results = self.test_clinical_assess_scenarios()
        
        # Test 2: Concurrent Requests
        print("\n2. Testing Concurrent Requests...")
        concurrent_results = self.test_concurrent_requests()
        
        # Test 3: Error Handling
        print("\n3. Testing Error Handling...")
        error_results = self.test_error_handling()
        
        # Compile summary
        total_tests = len(assessment_results) + len(error_results) + 1
        passed_tests = (
            sum(1 for r in assessment_results if r["passed"]) +
            sum(1 for r in error_results if r["passed"]) +
            (1 if concurrent_results["success_rate"] > 80 else 0)
        )
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests * 100,
            "assessment_results": assessment_results,
            "concurrent_results": concurrent_results,
            "error_results": error_results
        }
        
        print(f"\n=== Test Summary ===")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        return summary

# Usage
if __name__ == "__main__":
    tester = APITester()
    results = tester.run_full_test_suite()
    
    # Save results to file
    with open("api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Performance Testing

### Load Testing with Locust

```python
# locustfile.py
from locust import HttpUser, task, between
import json
import random

class CDSSUser(HttpUser):
    """Locust user for CDSS API performance testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize test data."""
        self.test_cases = [
            {
                "drugs": ["warfarin", "ibuprofen"],
                "age": 70,
                "gender": "Male",
                "allergies": [],
                "diagnosis": "Atrial Fibrillation and Osteoarthritis"
            },
            {
                "drugs": ["lisinopril", "ibuprofen"],
                "age": 34,
                "gender": "Female",
                "allergies": ["sulfa drugs"],
                "diagnosis": "Gestational Hypertension"
            },
            {
                "drugs": ["metformin", "naproxen"],
                "age": 55,
                "gender": "Male",
                "allergies": ["penicillin"],
                "diagnosis": "Type 2 Diabetes and Osteoarthritis"
            }
        ]
    
    @task(3)
    def clinical_assess(self):
        """Test clinical assessment endpoint."""
        test_case = random.choice(self.test_cases)
        
        with self.client.post(
            "/clinical-assess",
            json=test_case,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    # Verify required keys
                    if "Severity" in json_response and "References" in json_response:
                        response.success()
                    else:
                        response.failure("Missing required response keys")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def clinical_pathways(self):
        """Test clinical pathways endpoint."""
        test_case = random.choice(self.test_cases)
        request_data = {
            "input": test_case,
            "severity": "high"
        }
        
        with self.client.post(
            "/clinical-pathways",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if "clinical_pathways" in json_response:
                        pathways = json_response["clinical_pathways"]
                        if len(pathways) == 3:
                            response.success()
                        else:
                            response.failure("Incorrect number of pathways")
                    else:
                        response.failure("Missing clinical_pathways key")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

# Run with: locust -f locustfile.py --host=http://localhost:8000
```

### Performance Benchmarking

```python
# performance_benchmark.py
import time
import statistics
from typing import List, Dict
import requests
import json

class PerformanceBenchmark:
    """Performance benchmarking for CDSS API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def benchmark_endpoint(
        self, 
        endpoint: str, 
        data: Dict, 
        num_requests: int = 10
    ) -> Dict:
        """Benchmark a specific endpoint."""
        response_times = []
        success_count = 0
        
        print(f"Benchmarking {endpoint} with {num_requests} requests...")
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=data,
                    timeout=120
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                
                print(f"  Request {i+1}: {response_time:.2f}s (HTTP {response.status_code})")
                
            except requests.exceptions.Timeout:
                print(f"  Request {i+1}: TIMEOUT")
            except Exception as e:
                print(f"  Request {i+1}: ERROR - {e}")
        
        if response_times:
            benchmark_result = {
                "endpoint": endpoint,
                "num_requests": num_requests,
                "success_count": success_count,
                "success_rate": success_count / num_requests * 100,
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "response_times": response_times
            }
        else:
            benchmark_result = {
                "endpoint": endpoint,
                "num_requests": num_requests,
                "success_count": 0,
                "success_rate": 0,
                "error": "No successful requests"
            }
        
        return benchmark_result
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive performance benchmark."""
        print("=== CDSS Performance Benchmark ===\n")
        
        # Test cases for different endpoints
        test_cases = {
            "/clinical-assess": {
                "drugs": ["warfarin", "ibuprofen"],
                "age": 70,
                "gender": "Male",
                "allergies": [],
                "diagnosis": "Atrial Fibrillation and Osteoarthritis"
            },
            "/clinical-pathways": {
                "input": {
                    "drugs": ["metformin", "lisinopril"],
                    "age": 55,
                    "gender": "Female",
                    "allergies": ["penicillin"],
                    "diagnosis": "Diabetes and Hypertension"
                },
                "severity": "moderate"
            }
        }
        
        benchmark_results = {}
        
        for endpoint, data in test_cases.items():
            result = self.benchmark_endpoint(endpoint, data, num_requests=5)
            benchmark_results[endpoint] = result
            
            print(f"\nBenchmark Results for {endpoint}:")
            print(f"  Success Rate: {result.get('success_rate', 0):.1f}%")
            if 'avg_response_time' in result:
                print(f"  Average Response Time: {result['avg_response_time']:.2f}s")
                print(f"  Median Response Time: {result['median_response_time']:.2f}s")
                print(f"  Min/Max Response Time: {result['min_response_time']:.2f}s / {result['max_response_time']:.2f}s")
            print()
        
        return benchmark_results

# Usage
if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    with open("performance_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Integration Patterns

### Healthcare System Integration

```python
# healthcare_integration.py
from typing import Dict, List, Optional
import requests
import json
from dataclasses import dataclass
from enum import Enum

class IntegrationStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"

@dataclass
class PatientData:
    """Patient data structure for integration."""
    patient_id: str
    age: int
    gender: str
    current_medications: List[str]
    allergies: List[str]
    diagnoses: List[str]
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

@dataclass
class ClinicalDecision:
    """Clinical decision result structure."""
    patient_id: str
    assessment_id: str
    timestamp: str
    drug_interactions: List[str]
    recommendations: List[Dict]
    severity_level: str
    requires_physician_review: bool

class CDSSIntegrationClient:
    """Client for integrating CDSS with healthcare systems."""
    
    def __init__(self, cdss_base_url: str, api_key: Optional[str] = None):
        self.cdss_base_url = cdss_base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def assess_patient_medications(
        self, 
        patient: PatientData,
        new_medication: str
    ) -> ClinicalDecision:
        """Assess patient medications including new prescription."""
        
        # Prepare request data
        all_medications = patient.current_medications + [new_medication]
        primary_diagnosis = patient.diagnoses[0] if patient.diagnoses else ""
        
        request_data = {
            "drugs": all_medications,
            "age": patient.age,
            "gender": patient.gender,
            "allergies": patient.allergies,
            "diagnosis": primary_diagnosis
        }
        
        # Call CDSS API
        try:
            response = self.session.post(
                f"{self.cdss_base_url}/clinical-assess",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            
            assessment_result = response.json()
            
            # Get clinical pathways for high-risk scenarios
            pathways = None
            if assessment_result.get("Severity") == ["High"]:
                pathways_request = {
                    "input": request_data,
                    "severity": "high"
                }
                
                pathways_response = self.session.post(
                    f"{self.cdss_base_url}/clinical-pathways",
                    json=pathways_request,
                    timeout=60
                )
                
                if pathways_response.status_code == 200:
                    pathways = pathways_response.json()
            
            # Build clinical decision
            clinical_decision = ClinicalDecision(
                patient_id=patient.patient_id,
                assessment_id=f"assess_{int(time.time())}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                drug_interactions=assessment_result.get("Drug-Drug Interactions", []),
                recommendations=pathways.get("clinical_pathways", []) if pathways else [],
                severity_level=assessment_result.get("Severity", ["Unknown"])[0],
                requires_physician_review=assessment_result.get("Severity") == ["High"]
            )
            
            return clinical_decision
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"CDSS API error: {e}")
    
    def batch_assess_patients(
        self, 
        patients: List[PatientData],
        new_medications: Dict[str, str]
    ) -> List[ClinicalDecision]:
        """Batch assess multiple patients."""
        
        results = []
        
        for patient in patients:
            if patient.patient_id in new_medications:
                try:
                    decision = self.assess_patient_medications(
                        patient, 
                        new_medications[patient.patient_id]
                    )
                    results.append(decision)
                except Exception as e:
                    print(f"Error assessing patient {patient.patient_id}: {e}")
        
        return results

class EHRIntegration:
    """Electronic Health Record integration."""
    
    def __init__(self, ehr_api_url: str, cdss_client: CDSSIntegrationClient):
        self.ehr_api_url = ehr_api_url
        self.cdss_client = cdss_client
    
    def prescribe_medication_with_cdss(
        self, 
        patient_id: str, 
        medication: str,
        prescriber_id: str
    ) -> Dict:
        """Prescribe medication with CDSS validation."""
        
        # 1. Fetch patient data from EHR
        patient_data = self.fetch_patient_data(patient_id)
        
        # 2. Assess with CDSS
        clinical_decision = self.cdss_client.assess_patient_medications(
            patient_data, 
            medication
        )
        
        # 3. Handle based on severity
        if clinical_decision.requires_physician_review:
            return {
                "status": "requires_review",
                "message": "High-risk interaction detected. Physician review required.",
                "clinical_decision": clinical_decision,
                "prescription_id": None
            }
        else:
            # 4. Create prescription in EHR
            prescription_id = self.create_prescription(
                patient_id, 
                medication, 
                prescriber_id,
                clinical_decision
            )
            
            return {
                "status": "prescribed",
                "message": "Medication prescribed successfully.",
                "clinical_decision": clinical_decision,
                "prescription_id": prescription_id
            }
    
    def fetch_patient_data(self, patient_id: str) -> PatientData:
        """Fetch patient data from EHR system."""
        # Implementation would call EHR API
        # This is a mock implementation
        return PatientData(
            patient_id=patient_id,
            age=65,
            gender="Male",
            current_medications=["metformin", "lisinopril"],
            allergies=["penicillin"],
            diagnoses=["Type 2 Diabetes", "Hypertension"]
        )
    
    def create_prescription(
        self, 
        patient_id: str, 
        medication: str,
        prescriber_id: str,
        clinical_decision: ClinicalDecision
    ) -> str:
        """Create prescription in EHR system."""
        # Implementation would call EHR API
        # This is a mock implementation
        return f"rx_{patient_id}_{int(time.time())}"

# Usage Example
def main():
    # Initialize CDSS client
    cdss_client = CDSSIntegrationClient("http://localhost:8000")
    
    # Initialize EHR integration
    ehr_integration = EHRIntegration("http://ehr-api.hospital.com", cdss_client)
    
    # Example: Prescribe medication with CDSS check
    result = ehr_integration.prescribe_medication_with_cdss(
        patient_id="PAT001",
        medication="warfarin",
        prescriber_id="DR001"
    )
    
    print(f"Prescription result: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['clinical_decision']:
        decision = result['clinical_decision']
        print(f"Severity: {decision.severity_level}")
        print(f"Interactions: {len(decision.drug_interactions)}")

if __name__ == "__main__":
    main()
```

This comprehensive testing and integration guide provides:

1. **Complete test coverage** for all components
2. **Performance benchmarking** tools
3. **Integration patterns** for healthcare systems
4. **Real-world usage examples**
5. **Error handling validation**
6. **Concurrent testing scenarios**

The documentation enables developers to thoroughly test, validate, and integrate the CDSS system into existing healthcare workflows with confidence.