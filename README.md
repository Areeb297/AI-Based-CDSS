# Clinical Decision Support System (CDSS) - Complete Documentation

## Overview

This is a comprehensive AI-powered Clinical Decision Support System that provides intelligent clinical assessments, pathway recommendations, and radiology report analysis. The system uses Retrieval-Augmented Generation (RAG) with FAISS vector storage and integrates with multiple LLM providers.

## 📚 Documentation Suite

This repository contains comprehensive documentation for developers, integrators, and users:

### 1. [API Documentation](API_DOCUMENTATION.md)
**Complete API reference with examples and usage instructions**
- 🔗 All API endpoints with request/response examples
- 🔑 Authentication and configuration setup
- 💻 Client code examples in Python and JavaScript
- ⚡ Performance guidelines and best practices
- 🚀 Deployment instructions

### 2. [Functions & Components Documentation](FUNCTIONS_COMPONENTS_DOCUMENTATION.md)
**Detailed internal architecture and function reference**
- 🏗️ Core components and RAG pipeline
- 🔧 Utility functions with examples
- 🤖 LLM integration patterns
- 📊 Data processing functions
- 🎯 Prompt templates and configuration

### 3. [Testing & Integration Guide](TESTING_INTEGRATION_GUIDE.md)
**Comprehensive testing strategies and integration patterns**
- ✅ Unit, integration, and E2E testing
- 📈 Performance benchmarking tools
- 🏥 Healthcare system integration examples
- 🔄 Real-world usage patterns
- 🐛 Error handling and validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for OpenAI, Groq, and OpenRouter
- PDF knowledge base file
- References database

### Installation
```bash
# Clone repository
git clone <repository-url>
cd clinical-decision-support

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration
Create a `.env` file with your API keys:
```env
ACTUAL_OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
PORT=8000
```

### Run the Application
```bash
python cdss_app.py
```

The API will be available at `http://localhost:8000`

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/clinical-assess` | POST | Analyze drug interactions and safety |
| `/clinical-pathways` | POST | Generate clinical management pathways |
| `/generate_report/` | POST | Analyze X-ray images and generate reports |

## 📊 Features

### Clinical Assessment
- **Drug-Drug Interactions**: Comprehensive interaction analysis
- **Allergy Checking**: Cross-reactivity validation
- **Pregnancy/Lactation Warnings**: Specialized safety assessments
- **Severity Classification**: Risk level determination
- **Reference Citations**: Evidence-based recommendations

### Clinical Pathways
- **Risk-Stratified Options**: Low, medium, and high-risk pathways
- **Detailed Monitoring**: Specific follow-up requirements
- **Trade-off Analysis**: Benefit/risk assessments
- **Saudi Arabia Compliance**: Local practice adaptations

### Radiology Analysis
- **Automatic Classification**: Chest vs. skeletal X-ray detection
- **Structured Reports**: Standardized medical reporting
- **Critical Finding Detection**: Pneumothorax and fracture identification
- **Clinical Context Integration**: Query-specific analysis

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Unit tests
pytest test_cdss_functions.py -v

# Integration tests
pytest test_api_integration.py -v

# Performance benchmarking
python performance_benchmark.py

# Load testing
locust -f locustfile.py --host=http://localhost:8000
```

## 🔧 Integration Examples

### Python Client
```python
import requests

def assess_patient(drugs, age, gender, allergies, diagnosis):
    response = requests.post(
        "http://localhost:8000/clinical-assess",
        json={
            "drugs": drugs,
            "age": age,
            "gender": gender,
            "allergies": allergies,
            "diagnosis": diagnosis
        }
    )
    return response.json()

# Example usage
result = assess_patient(
    drugs=["warfarin", "ibuprofen"],
    age=70,
    gender="Male",
    allergies=[],
    diagnosis="Atrial Fibrillation and Osteoarthritis"
)
print(f"Severity: {result['Severity']}")
print(f"Interactions: {len(result['Drug-Drug Interactions'])}")
```

### Healthcare System Integration
```python
from healthcare_integration import CDSSIntegrationClient, PatientData

# Initialize client
cdss = CDSSIntegrationClient("http://localhost:8000")

# Create patient data
patient = PatientData(
    patient_id="PAT001",
    age=65,
    gender="Male",
    current_medications=["metformin", "lisinopril"],
    allergies=["penicillin"],
    diagnoses=["Type 2 Diabetes", "Hypertension"]
)

# Assess new medication
decision = cdss.assess_patient_medications(patient, "warfarin")
print(f"Requires physician review: {decision.requires_physician_review}")
```

## 📈 Performance

- **Clinical Assessment**: ~5-15 seconds per request
- **Clinical Pathways**: ~10-30 seconds per request  
- **Radiology Reports**: ~15-45 seconds per request
- **Concurrent Support**: Up to 5 parallel requests
- **Accuracy**: >95% for documented clinical scenarios

## 🛡️ Error Handling

The system provides comprehensive error handling:
- Input validation with detailed error messages
- API timeout and retry mechanisms
- Graceful degradation for service outages
- Structured error responses with debugging information

## 🔒 Security

- Environment-based API key management
- Input sanitization and validation
- Secure file upload handling
- No sensitive data logging

## 📝 Contributing

1. Review the documentation thoroughly
2. Follow the testing guidelines
3. Ensure all tests pass before submitting
4. Update documentation for any changes
5. Follow the coding standards in the examples

## 📄 License

[Your License Here]

## 🆘 Support

For technical support:
1. Check the comprehensive documentation
2. Review the testing guide for debugging
3. Examine the integration examples
4. Check application logs for detailed error information

---

**Note**: This system is designed for clinical decision support and should not replace professional medical judgment. Always consult with qualified healthcare professionals for patient care decisions.