# Clinical Decision Support System (CDSS) API Documentation

## Overview

The Clinical Decision Support System (CDSS) is a comprehensive FastAPI-based application that provides AI-powered clinical decision support through multiple specialized endpoints. The system uses Retrieval-Augmented Generation (RAG) with FAISS vector storage and integrates with multiple LLM providers including OpenAI, Groq, and OpenRouter.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Authentication & Configuration](#authentication--configuration)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Examples](#examples)
6. [Error Handling](#error-handling)
7. [Deployment](#deployment)

## System Architecture

### Core Components

- **FastAPI Application**: Modern web framework for building APIs
- **RAG Pipeline**: Retrieval-Augmented Generation using FAISS vector store
- **Multi-LLM Integration**: Support for OpenAI, Groq, and OpenRouter APIs
- **PDF Knowledge Base**: Clinical guidelines and reference materials
- **Vector Store**: FAISS-based similarity search for context retrieval

### Key Features

- Clinical drug interaction assessment
- Clinical pathway generation
- Radiology report analysis
- Reference citation management
- Structured output validation

## Authentication & Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI API (for embeddings)
ACTUAL_OPENAI_API_KEY=your_openai_api_key_here

# Groq API (for LLM inference)
GROQ_API_KEY=your_groq_api_key_here

# OpenRouter API (for specialized radiology analysis)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Custom port (defaults to 10000)
PORT=8000
```

### File Dependencies

- `CDSS_Expected_Outputs_Cases_ALL.pdf`: Clinical knowledge base
- `references_list_key_brackets_value_plain.json`: Reference citation database
- `faiss_cdss_pdf/`: FAISS vector index (auto-generated)

## API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Clinical Assessment Endpoint

**Endpoint:** `POST /clinical-assess`

**Description:** Analyzes drug combinations for interactions, contraindications, and safety warnings.

**Request Body:**
```json
{
  "drugs": ["warfarin", "ibuprofen"],
  "age": 70,
  "gender": "Male",
  "allergies": ["sulfa drugs"],
  "diagnosis": "Atrial Fibrillation and Osteoarthritis"
}
```

**Response Structure:**
```json
{
  "Drug-Drug Interactions": [
    "Risk of serious bleeding when warfarin is combined with NSAIDs like ibuprofen. [1][2]"
  ],
  "Drug-Allergy": [
    "No known cross-reactivity between warfarin/ibuprofen and sulfa drugs. [9]"
  ],
  "Drug-Disease Contraindications": [
    "NSAIDs may worsen cardiovascular risk in atrial fibrillation patients. [4]"
  ],
  "Ingredient Duplication": ["None"],
  "Pregnancy Warnings": ["None (not applicable in male patients). [3]"],
  "Lactation Warnings": ["None (not applicable in male patients). [3]"],
  "General Precautions": [
    "Monitor for bleeding signs when combining warfarin and NSAIDs. [1][2]"
  ],
  "Therapeutic Class Conflicts": [
    "Both affect clotting mechanisms through different pathways. [1][2]"
  ],
  "Warning Labels": [
    "Risk of serious bleeding, especially GI, when combining these medications. [1][7]"
  ],
  "Indications": [
    "Warfarin: Anticoagulant for stroke prevention in atrial fibrillation. [1]",
    "Ibuprofen: NSAID for pain and inflammation management. [2]"
  ],
  "Severity": ["High"],
  "References": [
    "[1] FDA label for warfarin: https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/009218s109lbl.pdf",
    "[2] FDA label for ibuprofen: https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf"
  ]
}
```

### 2. Clinical Pathways Endpoint

**Endpoint:** `POST /clinical-pathways`

**Description:** Generates three clinical management pathways (low, medium, high risk) for a given clinical scenario.

**Request Body:**
```json
{
  "input": {
    "drugs": ["warfarin", "ibuprofen"],
    "age": 70,
    "gender": "Male",
    "allergies": [],
    "diagnosis": "Atrial Fibrillation and Osteoarthritis"
  },
  "expected_output": {},
  "severity": "high"
}
```

**Response Structure:**
```json
{
  "clinical_pathways": [
    {
      "recommendation": "Acetaminophen 500mg orally every 6 hours as needed for pain, not exceeding 3g per day, for up to 10 days. Avoid NSAIDs entirely. [1][2]",
      "details": "Avoids NSAID-warfarin interaction, minimizing bleeding risk. [1][2]",
      "tradeoff": "Pain relief may be slower, but risk of major bleeding is minimized. [2][3]",
      "monitoring": "Check INR as per anticoagulation protocol. Review pain at 7 days. [2][4]"
    },
    {
      "recommendation": "Ibuprofen 200mg orally twice daily for up to 3 days with omeprazole 20mg daily for GI protection, under close monitoring. [2][3]",
      "details": "Short-term NSAID with GI protection offers controlled bleeding risk. [2][4]",
      "tradeoff": "Better symptom control but requires vigilant monitoring. [3][4]",
      "monitoring": "Check INR before, during, and after NSAID course; monitor for bleeding daily. [2][4]"
    },
    {
      "recommendation": "Ibuprofen 400mg three times daily for 7 days without protection. Do not use unless life-threatening situation and no alternatives.",
      "details": "High-dose NSAID in anticoagulated elderly carries very high bleeding risk.",
      "tradeoff": "Rapid pain relief but life-threatening complications risk.",
      "monitoring": "None."
    }
  ],
  "References": {
    "[1]": "FDA label for warfarin: https://www.accessdata.fda.gov/drugsatfda_docs/label/...",
    "[2]": "Saudi MOH Anticoagulation Guidelines: https://moh.gov.sa/guidelines.pdf"
  }
}
```

### 3. Radiology Report Generation Endpoint

**Endpoint:** `POST /generate_report/`

**Description:** Analyzes X-ray images and generates structured radiology reports with automatic image classification.

**Request Parameters:**
- `image` (file): X-ray image file (JPEG, PNG supported)
- `query` (form field, optional): Clinical context or specific questions

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/generate_report/" \
  -F "image=@chest_xray.jpg" \
  -F "query=65-year-old patient with chest pain"
```

**Response Structure (Chest X-ray):**
```json
{
  "View": "Frontal chest radiograph",
  "Pneumothorax": "Present. Small left apical pneumothorax.",
  "Findings": "Small left apical pneumothorax identified. Heart size normal. No acute infiltrates.",
  "Impression": "Small left apical pneumothorax requiring follow-up.",
  "Recommendations": "Follow-up chest X-ray in 24 hours to assess progression."
}
```

**Response Structure (Skeletal X-ray):**
```json
{
  "View": "AP and lateral views of the right hand",
  "Findings": "Non-displaced transverse fracture at the mid-shaft of the fourth metacarpal. No other acute fractures identified.",
  "Impression": "Right fourth metacarpal shaft fracture, non-displaced.",
  "Recommendations": "Orthopedic consultation recommended. Immobilization advised."
}
```

## Data Models

### ClinicalRequest
```python
class ClinicalRequest(BaseModel):
    drugs: list          # List of drug names
    age: int            # Patient age in years
    gender: str         # "Male" or "Female"
    allergies: list = []  # List of known allergies
    diagnosis: str = ''   # Primary diagnosis
```

### PathwayTestCase
```python
class PathwayTestCase(BaseModel):
    input: dict           # Clinical scenario parameters
    expected_output: dict = {}  # Optional expected outputs
    severity: str = ""    # Risk severity level
```

## Examples

### Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Example 1: Clinical Assessment
def assess_clinical_case():
    url = f"{BASE_URL}/clinical-assess"
    data = {
        "drugs": ["lisinopril", "ibuprofen"],
        "age": 34,
        "gender": "Female",
        "allergies": ["sulfa drugs"],
        "diagnosis": "Gestational Hypertension"
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("Clinical Assessment Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example 2: Clinical Pathways
def generate_pathways():
    url = f"{BASE_URL}/clinical-pathways"
    data = {
        "input": {
            "drugs": ["warfarin", "ibuprofen"],
            "age": 70,
            "gender": "Male",
            "allergies": [],
            "diagnosis": "Atrial Fibrillation and Osteoarthritis"
        },
        "severity": "high"
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("Clinical Pathways:")
        print(json.dumps(result, indent=2))

# Example 3: Radiology Report
def analyze_xray():
    url = f"{BASE_URL}/generate_report/"
    
    with open("chest_xray.jpg", "rb") as image_file:
        files = {"image": image_file}
        data = {"query": "65-year-old patient with chest pain"}
        
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            print("Radiology Report:")
            print(json.dumps(result, indent=2))

# Run examples
if __name__ == "__main__":
    assess_clinical_case()
    generate_pathways()
    # analyze_xray()  # Uncomment when you have an image file
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const BASE_URL = 'http://localhost:8000';

// Clinical Assessment Example
async function assessClinicalCase() {
    try {
        const response = await axios.post(`${BASE_URL}/clinical-assess`, {
            drugs: ['metformin', 'naproxen'],
            age: 55,
            gender: 'Male',
            allergies: ['penicillin'],
            diagnosis: 'Type 2 Diabetes Mellitus and Osteoarthritis'
        });
        
        console.log('Clinical Assessment Result:');
        console.log(JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Radiology Report Example
async function analyzeXray() {
    try {
        const form = new FormData();
        form.append('image', fs.createReadStream('xray_image.jpg'));
        form.append('query', 'Patient with suspected fracture');
        
        const response = await axios.post(`${BASE_URL}/generate_report/`, form, {
            headers: form.getHeaders()
        });
        
        console.log('Radiology Report:');
        console.log(JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Run examples
assessClinicalCase();
```

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Invalid input parameters",
  "details": "Age must be a positive integer"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Error during RAG chain invocation",
  "details": "Failed to connect to LLM service",
  "raw_output": null
}
```

**API Key Missing:**
```json
{
  "error": "GROQ_API_KEY not found in environment variables"
}
```

### Best Practices

1. **Always validate input data** before sending requests
2. **Handle network timeouts** - some requests may take 10-30 seconds
3. **Check response status codes** before processing results
4. **Store API keys securely** using environment variables
5. **Implement retry logic** for transient errors

## Rate Limits and Performance

- **Clinical Assessment**: ~5-15 seconds per request
- **Clinical Pathways**: ~10-30 seconds per request  
- **Radiology Reports**: ~15-45 seconds per request
- **Concurrent Requests**: Limited by LLM API quotas
- **File Upload Limit**: 10MB for images

## Deployment

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd clinical-decision-support

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
python cdss_app.py
```

### Production Deployment

```bash
# Using uvicorn directly
uvicorn cdss_app:app --host 0.0.0.0 --port 8000 --workers 1

# Using Docker
docker build -t cdss-api .
docker run -p 8000:8000 --env-file .env cdss-api
```

### Environment Setup Checklist

- [ ] Install Python 3.8+
- [ ] Install dependencies from requirements.txt
- [ ] Set up API keys in .env file
- [ ] Place PDF knowledge base in correct location
- [ ] Verify FAISS index builds successfully
- [ ] Test all endpoints with sample data

## Support and Troubleshooting

### Common Issues

1. **FAISS Index Build Fails**
   - Ensure PDF file exists and is readable
   - Check ACTUAL_OPENAI_API_KEY is valid
   - Verify sufficient disk space

2. **LLM API Errors**
   - Validate all API keys are current
   - Check API quota limits
   - Verify network connectivity

3. **Memory Issues**
   - Large PDFs may require more RAM
   - Consider chunking large documents
   - Monitor vector store size

### Debug Mode

Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

For additional support, check the application logs and verify all environment variables are properly configured.